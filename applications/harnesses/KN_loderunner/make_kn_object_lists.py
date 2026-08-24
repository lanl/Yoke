"""Create a seeded, object-level train/val/test split for the 9-band KN data.

The kilonova pipeline can train on the SAME objects viewed two ways: a
"realistic" light-curve set (sparse, upper limits dropped) and a "dense" set
(same objects, denser cadence, no limiting-mag cut). The two views live in
separate directories but share filename stems (``lc_<id>``). To avoid a single
object leaking across the split (its realistic view in train, its dense view in
test, or vice versa), the split is computed ONCE at the object level and applied
to BOTH directories by stem.

This script writes three stem lists (one object identifier per line) --

    kn_rubin_ztf_train.txt
    kn_rubin_ztf_val.txt
    kn_rubin_ztf_test.txt

-- to ``applications/filelists/`` by default. The training harness reads each
into a set and passes it as ``object_ids`` to
``Kilonova_lc_scalar_context_DataSet_9band`` for both the realistic and dense
directories.

Determinism: the split uses a seeded ``numpy.random.default_rng`` and sorts the
stems BEFORE shuffling, so the result is reproducible across machines regardless
of filesystem ``glob`` ordering. Run it once and commit the three lists.

Example:
    python make_kn_object_lists.py \
        --realistic_glob "/path/to/rubin_ztf_10000_dataset/lc_*.npz" \
        --dense_glob "/path/to/rubin_ztf_10000_dense/lc_*.npz" \
        --seed 20240817
"""

import argparse
import glob
import os


def _stem(path: str) -> str:
    """Return the object identifier: filename without directory or extension."""
    return os.path.splitext(os.path.basename(path))[0]


def make_object_split(
    realistic_glob: str,
    seed: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    dense_glob: str = None,
) -> tuple[list[str], list[str], list[str]]:
    """Split object stems into train/val/test deterministically.

    The split is computed over the REALISTIC universe of objects (always
    present). The test fraction is the remainder, so no object is dropped.

    Args:
        realistic_glob (str): Glob for the realistic light-curve files.
        seed (int): Seed for the shuffle RNG.
        train_frac (float): Fraction of objects for training.
        val_frac (float): Fraction of objects for validation. Test = remainder.
        dense_glob (str): Optional glob for the dense files. Only used to report
            how many realistic stems are (not) covered by the dense set.

    Returns:
        (train, val, test): three sorted lists of object stems.
    """
    import numpy as np

    realistic_stems = sorted({_stem(f) for f in glob.glob(realistic_glob)})
    if not realistic_stems:
        raise ValueError(f"No files matched realistic_glob: {realistic_glob!r}")

    if dense_glob is not None:
        dense_stems = {_stem(f) for f in glob.glob(dense_glob)}
        missing = set(realistic_stems) - dense_stems
        extra = dense_stems - set(realistic_stems)
        print(
            f"Dense coverage: {len(dense_stems)} dense stems; "
            f"{len(missing)} realistic objects have NO dense counterpart; "
            f"{len(extra)} dense-only stems (ignored)."
        )

    # Sort-then-shuffle: deterministic regardless of glob/filesystem ordering.
    stems = sorted(realistic_stems)
    rng = np.random.default_rng(seed)
    rng.shuffle(stems)

    n = len(stems)
    n_train = int(np.floor(train_frac * n))
    n_val = int(np.floor(val_frac * n))

    train = sorted(stems[:n_train])
    val = sorted(stems[n_train:n_train + n_val])
    test = sorted(stems[n_train + n_val:])  # remainder, no dropped objects

    return train, val, test


def _write_list(path: str, stems: list[str]) -> None:
    """Write one stem per line to path."""
    with open(path, "w") as fh:
        for s in stems:
            fh.write(s + "\n")
    print(f"Wrote {len(stems):6d} stems -> {path}")


def main() -> None:
    """Parse args, build the split, and write the three stem lists."""
    here = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.abspath(
        os.path.join(here, "..", "..", "filelists")
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--realistic_glob",
        type=str,
        default=(
            "/net/sescratch1/atoivonen/data/KN_lightcurves/"
            "rubin_ztf_10000_dataset_same_seed/lc_*.npz"
        ),
        help="Glob for the realistic light-curve files.",
    )
    parser.add_argument(
        "--dense_glob",
        type=str,
        default=(
            "/net/sescratch1/atoivonen/data/KN_lightcurves/"
            "rubin_ztf_dense_10000_dataset_same_seed/lc_*.npz"
        ),
        help="Optional glob for the dense files (for coverage reporting only).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=default_out,
        help="Directory to write the stem lists into.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="kn_rubin_ztf",
        help="Filename prefix for the three output lists.",
    )
    parser.add_argument("--seed", type=int, default=20240817)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    args = parser.parse_args()

    train, val, test = make_object_split(
        realistic_glob=args.realistic_glob,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        dense_glob=args.dense_glob,
    )

    total = len(train) + len(val) + len(test)
    print(
        f"Split {total} objects (seed={args.seed}): "
        f"{len(train)} train / {len(val)} val / {len(test)} test"
    )

    os.makedirs(args.out_dir, exist_ok=True)
    _write_list(os.path.join(args.out_dir, f"{args.prefix}_train.txt"), train)
    _write_list(os.path.join(args.out_dir, f"{args.prefix}_val.txt"), val)
    _write_list(os.path.join(args.out_dir, f"{args.prefix}_test.txt"), test)


if __name__ == "__main__":
    main()
