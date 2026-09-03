"""Dataset for DiffusionLodeRunner temporal prediction.

This module provides dataset classes for training score-based diffusion models
on temporal prediction tasks. The datasets handle loading sequential frames,
applying forward diffusion, and preparing conditioning information.
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from yoke.datasets.lsc_dataset import LSCread_npz_NaN, volfrac_density
from yoke.utils.diffusion.noise_schedulers import VPCosineNoiseSchedule


class DiffusionLSC_temporal_DataSet(Dataset):
    """Temporal dataset for DiffusionLSC training.

    This dataset is for multi-channel images at two different times from
    the *Perturned Layer Interface* simulation.
    The channels in the images returned are the densities for
    each material at a given time as well as the (R, Z)-velocity
    fields.

    The dataset finds pairs of images seperated by lead time delta_t, denoted (x, y)
    and applies the forward diffusion process specified by the diffusion time tau
    to y to produce y_tau and noise.

    The dataset returns the tuple (x, y_tau, noise, delta_t, tau).

    """

    def __init__(
        self,
        LSC_NPZ_DIR: str,
        file_prefix_list: str,
        max_timeIDX_offset: int,
        max_file_checks: int,
        half_image: bool = True,
        in_vars: np.array = np.array(
            [
                "density_case",
                "density_cushion",
                "density_maincharge",
                "density_outside_air",
                "density_striker",
                "density_throw",
                "Uvelocity",
                "Wvelocity",
            ]
        ),
        out_vars: np.array = None,
        noise_schedule: VPCosineNoiseSchedule = None,
    ) -> None:
        """Initialize DiffusionLSC temporal dataset.

        Args:
            LSC_NPZ_DIR (str): Location of LSC NPZ files.
            file_prefix_list (str): Text file listing unique prefixes corresponding
                                    to unique simulations.
            max_timeIDX_offset (int): Maximum timesteps-ahead to attempt
                                      prediction for. A prediction image will be chosen
                                      within this timeframe at random.
            max_file_checks (int): This dataset generates two random time indices and
                                   checks if the corresponding files exist. This
                                   argument controls the maximum number of times indices
                                   are generated before throwing an error.
            half_image (bool): If True then returned images are NOT reflected about axis
                               of symmetry and half-images are returned instead.
            in_vars (np.array, optional): Array of hydro field names for the input x.
                                               Defaults to:
                                               [
                                                   "density_case",
                                                   "density_cushion",
                                                   "density_maincharge",
                                                   "density_outside_air",
                                                   "density_striker",
                                                   "density_throw",
                                                   "Uvelocity",
                                                   "Wvelocity",
                                               ].
            out_vars (np.array, optional): Array of hydro field names for the output y.
                                               If None, uses same fields as in_vars.
            noise_schedule (VPCosineNoiseSchedule, optional): Noise scheduler
                                                            for forward diffusion.
        """
        # Model Arguments
        self.LSC_NPZ_DIR = LSC_NPZ_DIR
        self.max_timeIDX_offset = max_timeIDX_offset
        self.max_file_checks = max_file_checks
        self.half_image = half_image
        self.in_vars = in_vars
        self.out_vars = out_vars if out_vars is not None else in_vars
        self.noise_schedule = noise_schedule or VPCosineNoiseSchedule()

        # Create filelist
        with open(file_prefix_list) as f:
            self.file_prefix_list = [line.rstrip() for line in f]

        # Shuffle the list of prefixes in-place
        random.shuffle(self.file_prefix_list)

        self.Nsamples = len(self.file_prefix_list)

        # Initialize random number generator for time index selection
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        """Return effectively infinite number of samples in dataset."""
        return int(1e6)

    def _load_hydro_fields(
        self, npz_file: np.lib.npyio.NpzFile, field_list: list[str]
    ) -> torch.Tensor:
        """Load and process hydro fields from NPZ file.

        Args:
            npz_file: Loaded NPZ file.
            field_list: List of field names to extract.

        Returns:
            Tensor of shape (C, H, W) with stacked fields.
        """
        field_imgs = []
        for hfield in field_list:
            tmp_img = LSCread_npz_NaN(npz_file, hfield)
            # Reweight densities by volume fraction
            tmp_img = volfrac_density(tmp_img, npz_file, hfield)

            # Reflect image if not half_image
            if not self.half_image:
                tmp_img = np.concatenate((np.fliplr(tmp_img), tmp_img), axis=1)

            field_imgs.append(tmp_img)

        # Stack fields channel-first
        return torch.tensor(np.stack(field_imgs, axis=0), dtype=torch.float32)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (x, y_tau, noise, delta_t, tau).

        Args:
            index: Dataset index.

        Returns:
            - x: Conditioning input of shape (B, C_in, H, W)
                    images at time t
            - y_tau: Noised target of shape (B, C_out, H, W)
                    images at time t + delta_t (t + Δt)
                    with diffusion noise at 'time' tau (τ) applied
            - noise: The noise that was added of shape (B, C_out, H, W)
            - lead_times: shape (B,) scalar tensor
                Time offset between frames delta_t (Δt)
            - tau: shape (B,) scalar tensor
                    Diffusion time in [0, 1]
        """
        # Rotate index if necessary
        index = index % self.Nsamples

        # Get the input image. Try several indices if necessary.
        prefix_attempt = 0
        prefix_loop_break = False
        while prefix_attempt < 5:
            file_prefix = self.file_prefix_list[index]

            # Use `while` loop to search until a pair of files which exists is found.
            attempt = 0
            while attempt < self.max_file_checks:
                # Files have name format *lsc240420_id01001_pvi_idx00000.npz*.
                #
                # Choose random starting index 0-(100-max_timeIDX_offset) so
                # the end index will be less than or equal to 99.
                seqLen = self.rng.integers(0, self.max_timeIDX_offset, endpoint=True)
                startIDX = self.rng.integers(0, 100 - seqLen, endpoint=True)
                endIDX = startIDX + seqLen

                # Construct file names
                start_file = file_prefix + f"_pvi_idx{startIDX:05d}.npz"
                end_file = file_prefix + f"_pvi_idx{endIDX:05d}.npz"

                # Check if both files exist
                start_file_path = Path(self.LSC_NPZ_DIR + start_file)
                end_file_path = Path(self.LSC_NPZ_DIR + end_file)

                if start_file_path.is_file() and end_file_path.is_file():
                    prefix_loop_break = True
                    break

                attempt += 1

            if attempt == self.max_file_checks:
                fnf_msg = (
                    "In DiffusionLSC_temporal_DataSet, "
                    "max_file_checks "
                    f"reached for prefix: {file_prefix}"
                )
                print(fnf_msg, file=sys.stderr)

            # Break outer loop if time-pairs were found.
            if prefix_loop_break:
                break

            # Try different prefix if no time-pairs are found.
            print(
                f"Prefix attempt {prefix_attempt + 1} failed. Trying next prefix.",
                file=sys.stderr,
            )
            prefix_attempt += 1
            index = (index + 1) % self.Nsamples  # Rotate index if necessary

        # Load NPZ files. Raise exceptions if file is not able to be loaded.
        try:
            start_npz = np.load(self.LSC_NPZ_DIR + start_file)
        except Exception as e:
            print(
                f"Error loading start file: {self.LSC_NPZ_DIR + start_file}",
                file=sys.stderr,
            )
            raise e

        try:
            end_npz = np.load(self.LSC_NPZ_DIR + end_file)
        except Exception as e:
            print(
                f"Error loading end file: {self.LSC_NPZ_DIR + end_file}",
                file=sys.stderr,
            )
            start_npz.close()
            raise e

        # Load conditioning input (current frame)
        x = self._load_hydro_fields(start_npz, self.in_vars)

        # Load clean target (future frame)
        y = self._load_hydro_fields(end_npz, self.out_vars)

        # Calculate lead time (time offset between frames)
        lead_time = torch.tensor(0.25 * (endIDX - startIDX), dtype=torch.float32)

        # Close the npzs
        start_npz.close()
        end_npz.close()

        # Sample diffusion time uniformly from [0, 1]
        tau = torch.rand(1).item()  # Scalar value
        tau_tensor = torch.tensor(tau, dtype=torch.float32)

        # Apply forward diffusion to get noised target
        y_tau, noise = self.noise_schedule.forward_diffusion(
            y.unsqueeze(0), tau_tensor.unsqueeze(0)
        )

        # Remove batch dimension added for forward_diffusion
        y_tau = y_tau.squeeze(0)
        noise = noise.squeeze(0)

        return x, y_tau, noise, lead_time, tau_tensor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test DiffusionLSC_temporal_DataSet functionality"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing NPZ files",
    )
    parser.add_argument(
        "--file_prefix_list",
        type=str,
        required=True,
        help="Text file with list of file prefixes",
    )
    parser.add_argument(
        "--max_timeIDX_offset",
        type=int,
        default=10,
        help="Maximum time index offset (default: 10)",
    )
    parser.add_argument(
        "--max_file_checks",
        type=int,
        default=100,
        help="Maximum file check attempts (default: 100)",
    )
    parser.add_argument(
        "--half_image",
        action="store_true",
        help="Use half images (no reflection)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to test (default: 5)",
    )
    parser.add_argument(
        "--in_vars",
        type=str,
        nargs="+",
        default=None,
        help="Input variable names (space-separated)",
    )
    parser.add_argument(
        "--out_vars",
        type=str,
        nargs="+",
        default=None,
        help="Output variable names (space-separated)",
    )

    args = parser.parse_args()

    # Convert variable lists to numpy arrays if provided
    in_vars = (
        np.array(args.in_vars)
        if args.in_vars
        else [
            "density_case",
            "density_cushion",
            "density_maincharge",
            "density_outside_air",
            "density_striker",
            "density_throw",
            "Uvelocity",
            "Wvelocity",
        ]
    )
    out_vars = (
        np.array(args.out_vars)
        if args.out_vars
        else [
            "density_case",
            "density_cushion",
            "density_maincharge",
            "density_outside_air",
            "density_striker",
            "density_throw",
            "Uvelocity",
            "Wvelocity",
        ]
    )

    # Create dataset
    print("Creating dataset...")
    dataset = DiffusionLSC_temporal_DataSet(
        LSC_NPZ_DIR=args.data_dir,
        file_prefix_list=args.file_prefix_list,
        max_timeIDX_offset=args.max_timeIDX_offset,
        max_file_checks=args.max_file_checks,
        half_image=args.half_image,
        in_vars=in_vars,
        out_vars=out_vars,
    )

    print(f"Dataset created with {dataset.Nsamples} file prefixes")
    print(f"Dataset length: {len(dataset)}")
    print(f"Input variables: {dataset.in_vars}")
    print(f"Output variables: {dataset.out_vars}")
    print(f"Half image mode: {dataset.half_image}")
    print()

    # Test loading samples
    print(f"Testing {args.num_samples} samples...")
    for i in range(args.num_samples):
        try:
            x, y_tau, noise, lead_time, tau = dataset[i]
            print(f"\nSample {i}:")
            print(f"  x shape: {x.shape}")
            print(f"  y_tau shape: {y_tau.shape}")
            print(f"  noise shape: {noise.shape}")
            print(f"  lead_time: {lead_time.item():.4f}")
            print(f"  tau: {tau.item():.4f}")
            print(f"  x range: [{x.min().item():.4f}, {x.max().item():.4f}]")
            print(f"  y_tau range: [{y_tau.min().item():.4f}, {y_tau.max().item():.4f}]")
            print(f"  noise range: [{noise.min().item():.4f}, {noise.max().item():.4f}]")
        except Exception as e:
            print(f"\nError loading sample {i}: {e}")
            import traceback

            traceback.print_exc()

    print("\nDataset test complete!")
