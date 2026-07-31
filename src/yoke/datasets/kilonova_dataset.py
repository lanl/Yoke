"""Relating to kilonova light-curve data.

Functions and classes for torch DataSets which sample kilonova light-curve
(g/r/i band) data, along with helpers for computing and caching per-band
normalization statistics.

"""

####################################
# Packages
####################################
import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


EPS = 1e-6


def compute_band_normalization(
    file_prefix_list: list[str],
    band_keys: tuple[str, ...] = ("arr_ztfg", "arr_ztfr", "arr_ztfi"),
    value_col: int = 1,
    stats_path: str = "kilonova_gri_norm_stats.npz",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute global per-band mean/std over the training files only.

    Args:
        file_prefix_list (list[str]): List of npz files to accumulate stats over.
        band_keys (tuple[str, ...]): Keys of the bands to normalize.
        value_col (int): Column index of the value to accumulate per band.
        stats_path (str): Path to save the computed statistics to.

    Returns:
        means (np.ndarray): Per-band means, shape [n_bands].
        stds (np.ndarray): Per-band standard deviations, shape [n_bands].
    """
    sums = np.zeros(len(band_keys), dtype=np.float64)
    sums_sq = np.zeros(len(band_keys), dtype=np.float64)
    counts = np.zeros(len(band_keys), dtype=np.float64)

    for fn in file_prefix_list:
        data = np.load(fn, allow_pickle=True)

        for b, key in enumerate(band_keys):
            vals = data[key][:, value_col].astype(np.float64)

            finite = np.isfinite(vals)
            vals = vals[finite]

            sums[b] += vals.sum()
            sums_sq[b] += np.square(vals).sum()
            counts[b] += vals.size

        data.close()

    means = sums / counts
    variances = sums_sq / counts - means**2
    variances = np.maximum(variances, 1e-12)
    stds = np.sqrt(variances)

    means = means.astype(np.float32)
    stds = stds.astype(np.float32)

    np.savez(
        stats_path,
        means=means,
        stds=stds,
        band_keys=np.array(band_keys),
        value_col=value_col,
    )

    print("Saved normalization stats:", stats_path)
    print("means:", means)
    print("stds:", stds)

    return means, stds


def load_or_compute_band_normalization(
    stats_path: str = "kilonova_gri_norm_stats.npz",
    band_keys: tuple[str, ...] = ("arr_ztfg", "arr_ztfr", "arr_ztfi"),
    value_col: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Load cached per-band normalization stats or compute them if missing.

    Args:
        stats_path (str): Path to load/save the statistics.
        band_keys (tuple[str, ...]): Keys of the bands to normalize.
        value_col (int): Column index of the value to accumulate per band.

    Returns:
        means (np.ndarray): Per-band means, shape [n_bands].
        stds (np.ndarray): Per-band standard deviations, shape [n_bands].
    """
    # FIXME: hardcoded scratch path. Should be passed in as an argument so this
    # library function does not depend on a user-specific filesystem location.
    file_prefix_list = sorted(
        glob.glob(
            "/net/sescratch1/atoivonen/data/KN_lightcurves/uniform_dataset_20000/lc_*.npz"
        )
    )

    if os.path.exists(stats_path):
        stats = np.load(stats_path, allow_pickle=True)
        means = stats["means"].astype(np.float32)
        stds = stats["stds"].astype(np.float32)
        stats.close()

        print("Loaded normalization stats:", stats_path)
        print("means:", means)
        print("stds:", stds)

        return means, stds

    return compute_band_normalization(
        file_prefix_list=file_prefix_list,
        band_keys=band_keys,
        value_col=value_col,
        stats_path=stats_path,
    )


class Kilonova_lc_scalar_context_DataSet_gri(Dataset):
    """Scalar-context kilonova light-curve dataset for g/r/i bands.

    Each sample provides a flattened window of normalized per-band values plus
    relative observation times as input, and either the normalized next value
    or the normalized delta as the target.
    """

    def __init__(
        self,
        N_imgs: int = 0,
        context_len: int = 5,
        band_keys: tuple[str, ...] = ("arr_ztfg", "arr_ztfr", "arr_ztfi"),
        value_col: int = 1,
        means: np.ndarray = None,
        stds: np.ndarray = None,
        predicts_delta: bool = True,
    ) -> None:
        """Initialize the dataset and build the sample index.

        Args:
            N_imgs (int): Number of light-curve files to sample; 0 uses all.
            context_len (int): Number of context timesteps per sample.
            band_keys (tuple[str, ...]): Keys of the bands to load.
            value_col (int): Column index of the value to load per band.
            means (np.ndarray): Per-band means for normalization, shape [n_bands].
            stds (np.ndarray): Per-band stds for normalization, shape [n_bands].
            predicts_delta (bool): If True target is the normalized delta,
                otherwise the normalized next value.
        """
        # FIXME: hardcoded scratch path. Should be passed in as an argument so
        # this dataset does not depend on a user-specific filesystem location.
        file_prefix_list = sorted(
            glob.glob(
                "/net/sescratch1/atoivonen/data/KN_lightcurves/"
                "uniform_dataset_20000/lc_*.npz"
            )
        )

        if N_imgs == 0:
            self.file_prefix_list = file_prefix_list
        else:
            self.file_prefix_list = list(
                np.random.choice(file_prefix_list, N_imgs, replace=False)
            )

        random.shuffle(self.file_prefix_list)

        self.context_len = context_len
        self.band_keys = tuple(band_keys)
        self.value_col = value_col
        self.n_channels = len(self.band_keys)
        self.predicts_delta = predicts_delta

        if means is None:
            raise ValueError(
                "means must be provided for per-band normalization. "
                "Expected shape [n_channels], e.g. [g_mean, r_mean, i_mean]."
            )

        if stds is None:
            raise ValueError(
                "stds must be provided for per-band normalization. "
                "Expected shape [n_channels], e.g. [g_std, r_std, i_std]."
            )

        self.means = np.asarray(means, dtype=np.float32)
        self.stds = np.asarray(stds, dtype=np.float32)

        if self.means.shape[0] != self.n_channels:
            raise ValueError(
                f"means has length {self.means.shape[0]}, "
                f"but n_channels={self.n_channels}"
            )

        if self.stds.shape[0] != self.n_channels:
            raise ValueError(
                f"stds has length {self.stds.shape[0]}, "
                f"but n_channels={self.n_channels}"
            )

        if np.any(self.stds <= 0):
            raise ValueError(f"All stds must be positive. Got stds={self.stds}")

        self.samples = []

        for file_idx, fn in enumerate(self.file_prefix_list):
            data = np.load(fn, allow_pickle=True)

            # Assume all bands share the same time grid.
            mjd = data[self.band_keys[0]][:, 0]
            n_times = len(mjd)

            data.close()

            max_start = n_times - context_len - 1
            for startIDX in range(max_start + 1):
                self.samples.append((file_idx, startIDX))

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the (input, target, Dt) triple for a given sample index.

        Args:
            index (int): Sample index.

        Returns:
            x (torch.Tensor): Flattened context values and relative times.
            target (torch.Tensor): Normalized next value or delta per band.
            Dt (torch.Tensor): Time delta to the target step.
        """
        file_idx, startIDX = self.samples[index]
        fn = self.file_prefix_list[file_idx]

        data = np.load(fn, allow_pickle=True)

        # Time grid from the first band.
        # Assumes arr_ztfg, arr_ztfr, arr_ztfi have matching MJD columns.
        mjd = data[self.band_keys[0]][:, 0].astype(np.float32)

        # Stack one scalar value column from each band.
        # vals shape: [T, 3] for g/r/i.
        vals = np.stack(
            [
                data[key][:, self.value_col].astype(np.float32)
                for key in self.band_keys
            ],
            axis=1,
        )

        data.close()

        # Per-band normalization.
        # vals[:, 0] = normalized g
        # vals[:, 1] = normalized r
        # vals[:, 2] = normalized i
        vals = (vals - self.means[None, :]) / (self.stds[None, :] + EPS)

        t0 = mjd.min()
        t_obs = mjd - t0

        target_idx = startIDX + self.context_len

        # Context values shape: [context_len, 3]
        context_vals = vals[startIDX:target_idx]

        # Relative observation times shape: [context_len]
        rel_t = t_obs[startIDX:target_idx] - t_obs[startIDX]
        rel_t = rel_t.astype(np.float32)

        # Flatten context as:
        # [g0, r0, i0, g1, r1, i1, ..., gK, rK, iK]
        context_flat = context_vals.reshape(-1).astype(np.float32)

        # Final input shape:
        # [(3 * context_len) + context_len]
        #
        # For context_len=5:
        # x.shape == [20]
        x = np.concatenate([context_flat, rel_t], axis=0)
        x = torch.tensor(x, dtype=torch.float32)

        if self.predicts_delta:
            # Predict normalized delta for each band:
            # [delta_g, delta_r, delta_i]
            target_vals = vals[target_idx] - vals[target_idx - 1]
        else:
            # Predict normalized absolute next value:
            # [g_next, r_next, i_next]
            target_vals = vals[target_idx]

        target = torch.tensor(target_vals, dtype=torch.float32)

        Dt = torch.tensor(
            t_obs[target_idx] - t_obs[target_idx - 1],
            dtype=torch.float32,
        )

        return x, target, Dt
