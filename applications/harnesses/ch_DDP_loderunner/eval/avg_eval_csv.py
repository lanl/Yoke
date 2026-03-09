"""Compute the statistics for MSE over evaluation CSV."""

import argparse
import pandas as pd


def main() -> None:
    """Compute and print."""
    # Parser CLI CSV-filename
    parser = argparse.ArgumentParser(
        description="Compute summary statistics of evaluation CSV."
        )
    parser.add_argument(
        '--csv',
        type=str,
        default='./testing_evaluation.csv',
        help="Path to evaluation CSV."
        )

    args = parser.parse_args()

    # Read the CSV file
    df = pd.read_csv(args.csv, header=None)

    # Extract the third column
    col = df[2]

    # Compute statistics of the third column
    mean_MSE = col.mean()
    p2_5, p50, p97_5 = col.quantile([0.025, 0.5, 0.975])

    # Print results
    print(f"Average MSE: {mean_MSE}")
    print(f"2.5-percentile: {p2_5}")
    print(f"50-percentile: {p50}")
    print(f"97.5-percentile: {p97_5}")


if __name__ == "__main__":
    main()
