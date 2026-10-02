"""Per-channel normalization statistics for LodeRunner-ViT LSC training.

Computes **global** (not per-frame), robust, scale-only (no shift) per-channel
normalization factors for the hydrodynamic fields used by the
``ch_ldrViT`` 2-frame harnesses
(:class:`yoke.datasets.lsc_dataset.LSC_rho2rho_temporal_2frame_DataSet`).

This is a *separate* script from
:mod:`applications.normalization.generate_lsc_normalization` on purpose: that
script computes per-time-bin average-density unbiasing and B-spline
design-parameter normalization for an unrelated geometry-to-final-state
problem. This script instead targets LodeRunner-ViT's variable-channel
image-to-image prediction and uses a different, channel-specific scheme.

The normalization scheme itself (classification, transforms, stats
computation, and NPZ save/load/apply) lives in
:mod:`yoke.datasets.lsc_normalization` so it can be reused directly by a
dataset class (e.g. a normalized sibling of
:class:`~yoke.datasets.lsc_dataset.LSC_rho2rho_temporal_2frame_DataSet`) and
covered by the standard ``src``-level test suite. This script only owns the
file-sampling/NPZ-scanning orchestration needed to *estimate* the statistics,
plus the CLI entry point. See
:mod:`yoke.datasets.lsc_normalization` for the full scheme description.

Output is a single NPZ keyed by channel name (see
:func:`yoke.datasets.lsc_normalization.save_channel_norm`). Intended usage
from within a dataset transform:
:func:`yoke.datasets.lsc_normalization.normalize_field`.
"""

import argparse

import numpy as np

from yoke.helpers import cli
from yoke.datasets.lsc_dataset import LSCread_npz_NaN, volfrac_density
from yoke.datasets.lsc_normalization import (
    DEFAULT_CHANNEL_LIST,
    compute_channel_stats,
    save_channel_norm,
)


def sample_file_indices(
    file_prefix_list: list[str],
    num_samples: int,
    max_time_idx: int,
    rng: np.random.Generator,
) -> list[tuple[str, int]]:
    """Randomly sample ``(prefix, time_idx)`` pairs for percentile estimation.

    Args:
        file_prefix_list (list[str]): Simulation prefixes (one per
            simulation, as in the ``lsc240420_prefixes_*.txt`` filelists).
        num_samples (int): Number of ``(prefix, time_idx)`` pairs to draw (with
            replacement across prefixes; without replacement is not required
            since this only feeds a statistical estimate).
        max_time_idx (int): Highest valid file time-index, inclusive, i.e. the
            index of the final suffix is in ``[0, max_time_idx]``.
        rng (np.random.Generator): Random number generator.

    Returns:
        list[tuple[str, int]]: Sampled ``(prefix, time_idx)`` pairs.
    """
    prefixes = rng.choice(file_prefix_list, size=num_samples, replace=True)
    time_idxs = rng.integers(0, max_time_idx, size=num_samples, endpoint=True)
    return list(zip(prefixes.tolist(), time_idxs.tolist()))


def collect_channel_samples(
    LSC_NPZ_DIR: str,
    channel_list: list[str],
    sampled_indices: list[tuple[str, int]],
) -> dict[str, list[np.ndarray]]:
    """Load sampled frames and collect raw per-channel voxel arrays.

    Missing files are skipped silently (the LSC simulation index range is not
    uniform across simulations); a warning is printed if a channel ends up with
    no samples at all.

    Args:
        LSC_NPZ_DIR (str): Directory containing the LSC NPZ files.
        channel_list (list[str]): Channel/field names to collect.
        sampled_indices (list[tuple[str, int]]): ``(prefix, time_idx)`` pairs
            from :func:`sample_file_indices`.

    Returns:
        dict[str, list[np.ndarray]]: Mapping from channel name to a list of
        raw (not normalized) 2-D voxel arrays, one per successfully loaded
        sample.
    """
    samples: dict[str, list[np.ndarray]] = {ch: [] for ch in channel_list}

    for prefix, time_idx in sampled_indices:
        filename = f"{prefix}_pvi_idx{time_idx:05d}.npz"
        filepath = LSC_NPZ_DIR + filename
        try:
            npz = np.load(filepath)
        except (FileNotFoundError, OSError):
            continue

        try:
            for channel in channel_list:
                field = LSCread_npz_NaN(npz, channel)
                field = volfrac_density(field, npz, channel)
                samples[channel].append(field)
        finally:
            npz.close()

    for channel, vals in samples.items():
        if not vals:
            print(
                f"[generate_lsc_channel_normalization] WARNING: no samples "
                f"collected for channel {channel!r}."
            )

    return samples


###################################################################
# CLI entry point
###################################################################
def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for this script.

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    descr_str = (
        "Compute global, scale-only, per-channel normalization statistics "
        "for LodeRunner-ViT LSC training (ch_ldrViT harnesses)."
    )
    parser = argparse.ArgumentParser(
        prog="Generate LSC Channel Normalization",
        description=descr_str,
        fromfile_prefix_chars="@",
    )
    parser = cli.add_filepath_args(parser)

    parser.add_argument(
        "--eval_filelist",
        action="store",
        type=str,
        default="lsc240420_prefixes_train_80pct.txt",
        help="Prefix filelist (one simulation prefix per line) to sample from.",
    )
    parser.add_argument(
        "--channel_list",
        nargs="+",
        type=str,
        default=DEFAULT_CHANNEL_LIST,
        help="Channel/field names to compute normalization statistics for.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2000,
        help="Number of (prefix, time_idx) frames to sample for the estimate.",
    )
    parser.add_argument(
        "--max_time_idx",
        type=int,
        default=99,
        help="Highest valid file time-index (inclusive) to sample from.",
    )
    parser.add_argument(
        "--density_percentile",
        type=float,
        default=99.9,
        help="Percentile of nonzero voxel magnitudes used as the one-sided "
        "density scale factor.",
    )
    parser.add_argument(
        "--low_percentile",
        type=float,
        default=0.1,
        help="Low percentile used for the symmetric velocity/log-range scale.",
    )
    parser.add_argument(
        "--high_percentile",
        type=float,
        default=99.9,
        help="High percentile used for the symmetric velocity/log-range scale.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for frame sampling.",
    )
    parser.add_argument(
        "--fileout",
        action="store",
        type=str,
        default="./lsc240420_channel_norm.npz",
        help="Output NPZ filepath for the computed normalization statistics.",
    )

    return parser


def main(args: argparse.Namespace) -> dict[str, dict[str, float | str]]:
    """Compute and save per-channel normalization statistics.

    Args:
        args (argparse.Namespace): Parsed command-line arguments (see
            :func:`build_parser`).

    Returns:
        dict[str, dict[str, float | str]]: The computed per-channel
        normalization records (also written to ``args.fileout``).
    """
    rng = np.random.default_rng(args.seed)

    eval_files = args.FILELIST_DIR + args.eval_filelist
    with open(eval_files) as f:
        file_prefix_list = [line.rstrip() for line in f if line.strip()]

    print(
        f"Sampling {args.num_samples} frames from {len(file_prefix_list)} "
        f"simulation prefixes..."
    )
    sampled_indices = sample_file_indices(
        file_prefix_list, args.num_samples, args.max_time_idx, rng
    )

    samples = collect_channel_samples(
        args.LSC_NPZ_DIR, args.channel_list, sampled_indices
    )

    stats: dict[str, dict[str, float | str]] = {}
    for channel in args.channel_list:
        pooled = np.concatenate([s.ravel() for s in samples[channel]])
        record = compute_channel_stats(
            channel,
            pooled,
            density_percentile=args.density_percentile,
            low_percentile=args.low_percentile,
            high_percentile=args.high_percentile,
        )
        stats[channel] = record
        print(
            f"  {channel:>20s}: family={record['family']:<10s} "
            f"scale={record['scale']:.6g}"
        )

    save_channel_norm(args.fileout, stats)
    print(f"Saved channel normalization statistics to {args.fileout}")

    return stats


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    main(cli_args)
