"""Plot raw and training-normalized L2-norm histograms for LSC channels."""

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

from yoke.datasets.lsc_normalization import load_channel_norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-channel raw and training-normalized L2-norm histograms "
            "from CSV files generated beside an LSC channel-normalization NPZ."
        )
    )
    parser.add_argument(
        "normalization_file",
        type=Path,
        help="Channel-normalization NPZ whose stem identifies the input CSV files.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: 50).",
    )
    return parser


def read_l2_norms(filepath: Path) -> tuple[list[tuple[str, int]], np.ndarray]:
    """Read sample identifiers and L2 norms from a normalization CSV."""
    with filepath.open(newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        required_columns = {"prefix", "time_idx", "l2_norm"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"CSV {filepath} must contain columns {sorted(required_columns)}."
            )

        sample_keys = []
        values = []
        for row in reader:
            sample_keys.append((row["prefix"], int(row["time_idx"])))
            values.append(float(row["l2_norm"]))

    norms = np.array(values, dtype=float)
    if norms.size == 0:
        raise ValueError(f"CSV {filepath} contains no L2-norm values.")
    if not np.all(np.isfinite(norms)):
        raise ValueError(f"CSV {filepath} contains non-finite L2-norm values.")
    return sample_keys, norms


def plot_channel_histograms(
    channel: str,
    raw_norms: np.ndarray,
    normalized_norms: np.ndarray,
    output_path: Path,
    bins: int,
) -> None:
    """Save raw and training-normalized histograms for one channel."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(raw_norms, bins=bins, color="tab:blue", edgecolor="black")
    axes[0].set_title("Unnormalized")
    axes[0].set_xlabel("L2 norm")
    axes[0].set_ylabel("Sample count")

    axes[1].hist(normalized_norms, bins=bins, color="tab:orange", edgecolor="black")
    axes[1].set_title("Training normalized")
    axes[1].set_xlabel("L2 norm")
    axes[1].set_ylabel("Sample count")

    figure.suptitle(channel)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main(args: argparse.Namespace) -> None:
    """Create per-channel and channel-average two-panel histogram PNGs."""
    if args.bins <= 0:
        raise ValueError("--bins must be greater than zero.")

    normalization_file = args.normalization_file.resolve()
    stats = load_channel_norm(str(normalization_file))
    if not stats:
        raise ValueError(f"Normalization file {normalization_file} has no channels.")

    stem = normalization_file.stem
    reference_keys: list[tuple[str, int]] | None = None
    raw_channel_norms = []
    normalized_channel_norms = []

    for channel, record in stats.items():
        raw_path = normalization_file.with_name(f"{stem}_{channel}_l2.csv")
        raw_keys, raw_norms = read_l2_norms(raw_path)
        if reference_keys is None:
            reference_keys = raw_keys
        elif raw_keys != reference_keys:
            raise ValueError(
                f"CSV {raw_path} does not contain the same ordered samples as "
                "other channels."
            )

        if record["family"] == "log_range":
            scaled_source_path = normalization_file.with_name(
                f"{stem}_{channel}_signed_log1p_l2.csv"
            )
            scaled_source_keys, scaled_source_norms = read_l2_norms(scaled_source_path)
            if scaled_source_keys != reference_keys:
                raise ValueError(
                    f"CSV {scaled_source_path} does not contain the same ordered "
                    "samples as the raw channel CSV."
                )
        else:
            scaled_source_norms = raw_norms

        normalized_norms = scaled_source_norms / float(record["scale"])
        raw_channel_norms.append(raw_norms)
        normalized_channel_norms.append(normalized_norms)
        output_path = normalization_file.with_name(f"{stem}_{channel}_l2_histograms.png")
        plot_channel_histograms(
            channel, raw_norms, normalized_norms, output_path, args.bins
        )
        print(f"Saved {output_path}")

    raw_average = np.mean(np.stack(raw_channel_norms), axis=0)
    normalized_average = np.mean(np.stack(normalized_channel_norms), axis=0)
    average_output_path = normalization_file.with_name(
        f"{stem}_channel_average_l2_histograms.png"
    )
    plot_channel_histograms(
        "Average across channels",
        raw_average,
        normalized_average,
        average_output_path,
        args.bins,
    )
    print(f"Saved {average_output_path}")


if __name__ == "__main__":
    main(build_parser().parse_args())
