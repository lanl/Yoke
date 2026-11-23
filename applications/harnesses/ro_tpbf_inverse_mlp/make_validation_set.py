"""Design a Truchas validation set for the weld-planning recipe models.

The weld-planning inverse maps (desired depth, beam sigma[, S wt%]) to laser
(power, scan speed). The honest test is to run each model's **recipe** in
Truchas at the material's true S wt% and compare the depth Truchas produces
with the depth that was asked for. This script designs that test:

1. Draws target welds (desired ``depth_um``, ``sigma_m``, ``S_wpct``) with a
   seeded Latin hypercube: depth uniform over the 10th-90th percentile of the
   melted simulations (inside the data, away from the melting threshold), sigma
   uniform over its 10th-90th percentile, and log10 S uniform over the full
   sampled S range.
2. Keeps a target only if it lies inside the data's support (its nearest
   training point, in z-scored (depth, sigma, log10 S), is no farther than the
   90th percentile of the training points' own nearest-neighbor distances), and
   if every model's recipe stays within the trained power and speed ranges, so
   the test measures the models rather than extrapolation.
3. For each kept target, writes four recipes:

   - ``MLP_withS``: the mean of the 10 k-fold planning MLPs (studies 61-70);
   - ``quadratic_withS``: quadratic least squares fitted on all melted rows;
   - ``MLP_noS``: the mean of the 10 S-free planning MLPs (studies 71-80);
   - ``quadratic_noS``: quadratic least squares without S, all melted rows.

   S-free recipes depend only on (depth, sigma); the run still uses the
   target's true S, because the material has an S content whether or not the
   model knew it.

4. Records, for every run, the depth that two forward surrogates expect
   (the mean of the 10 forward MLP v3 fold models, ``ro_tpbf_mlp`` studies
   31-40, and a quadratic fit on all melted rows). These let the same Truchas
   runs also validate the forward models on new inputs.

Outputs (in ``--outdir``):

- ``tpbf_validation_truchas_inputs.csv``: ``run_id, power_W, scan_speed_mps,
  sigma_m, S_wpct``, the Truchas inputs; all other inputs as in the training
  set (O_wpct 0, dsigma_source dgdT_mcnallan, t_final_s 1.5, ...);
- ``tpbf_validation_design.csv``: one row per run, with the target, the recipe
  source, and the surrogates' expected depths, to join with the Truchas
  results by ``run_id``.

Run from this harness directory::

    python make_validation_set.py --n_targets 25 --seed 20261120
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch

from yoke.datasets.tabular_csv import TabularCSV_DataSet
from yoke.models.mlp import SwiGLU_mlp, model_args_for_state_dict

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "tpbf_training_set_261105.csv")
FWD_DIR = os.path.join(HERE, "..", "ro_tpbf_mlp")
COMMON = {"require_true": ["melted"], "stratify_col": "stratum"}
RECIPE = ["power_W", "scan_speed_mps"]


def quad_features(x: np.ndarray) -> np.ndarray:
    """``[1, x, x_i x_j (i <= j)]`` design matrix for quadratic least squares.

    Args:
        x (np.ndarray): ``(n, d)`` inputs.

    Returns:
        np.ndarray: ``(n, 1 + d + d(d+1)/2)`` features.
    """
    d = x.shape[1]
    cols = [np.ones(len(x)), *x.T]
    cols += [x[:, i] * x[:, j] for i in range(d) for j in range(i, d)]
    return np.stack(cols, axis=1)


def latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` points of a ``d``-dimensional Latin hypercube in ``[0, 1)^d``.

    Args:
        n (int): Number of points.
        d (int): Number of dimensions.
        rng (np.random.Generator): Random generator.

    Returns:
        np.ndarray: ``(n, d)`` samples, one per stratum in every dimension.
    """
    u = (np.arange(n)[:, None] + rng.random((n, d))) / n
    for j in range(d):
        u[:, j] = rng.permutation(u[:, j])
    return u


def raw_inputs(targets: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Model inputs from target rows, with S as log10 (as in training).

    Args:
        targets (pd.DataFrame): Columns ``depth_um``, ``sigma_m``, ``S_wpct``
            (and, for forward models, the recipe columns).
        cols (list[str]): Input columns, in model order.

    Returns:
        np.ndarray: ``(n, len(cols))`` inputs.
    """
    x = targets[cols].to_numpy(dtype=float).copy()
    if "S_wpct" in cols:
        j = cols.index("S_wpct")
        x[:, j] = np.log10(x[:, j])
    return x


def quadratic_model(in_cols: list[str], out_cols: list[str], logs: list[str]) -> object:
    """Fit quadratic least squares on all melted rows; return a predictor.

    Args:
        in_cols (list[str]): Input columns.
        out_cols (list[str]): Output columns.
        logs (list[str]): Columns taken as log10 (as in the harness).

    Returns:
        Callable: ``f(targets) -> (n, len(out_cols))`` predictions in units.
    """
    # n_folds=2 with fold=0 is only a vehicle for the statistics: fit on the
    # union of both halves, i.e. every melted row.
    halves = [
        TabularCSV_DataSet(
            TABLE,
            in_cols,
            out_cols,
            split=s,
            n_folds=2,
            fold=0,
            log_cols=logs,
            normalize=False,
            **COMMON,
        )
        for s in ("train", "holdout")
    ]
    x = np.concatenate([h.inputs.double().numpy() for h in halves])
    y = np.concatenate([h.targets.double().numpy() for h in halves])
    xm, xs = x.mean(0), x.std(0)
    ym, ys = y.mean(0), y.std(0)
    w = np.linalg.lstsq(quad_features((x - xm) / xs), (y - ym) / ys, rcond=None)[0]

    def predict(targets: pd.DataFrame) -> np.ndarray:
        """Quadratic prediction for target rows."""
        z = (raw_inputs(targets, in_cols) - xm) / xs
        return (quad_features(z) @ w) * ys + ym

    return predict


def mlp_ensemble(harness: str, csv_name: str, train_script: str) -> object:
    """Load the 10 k-fold MLPs of one study CSV; return their mean predictor.

    Each fold model is applied with its own training-fold normalization (read
    from the study's input file), and predictions are averaged in units.

    Args:
        harness (str): Harness directory.
        csv_name (str): Study CSV in that harness.
        train_script (str): The harness's train script (its parser reads the
            study input files).

    Returns:
        Callable: ``f(targets) -> (n, n_out)`` mean prediction in units.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train", os.path.join(harness, train_script)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    studies = pd.read_csv(os.path.join(harness, csv_name), index_col=0, comment="#")
    members = []
    for sid in studies.index:
        sdir = os.path.join(harness, "runs", f"study_{sid:03d}")
        args = module.parser.parse_args([f"@{sdir}/study{sid:03d}_START.input"])
        tr = TabularCSV_DataSet(
            os.path.join(sdir, args.table_csv),
            args.input_cols,
            args.target_cols,
            split="train",
            n_folds=args.n_folds,
            fold=args.fold,
            split_seed=args.split_seed,
            require_true=args.require_true,
            stratify_col=args.stratify_col or None,
            log_cols=args.log_cols,
        )
        ckpt = sorted(
            glob.glob(os.path.join(sdir, f"study{sid:03d}_modelState_epoch*.pth"))
        )
        state = torch.load(ckpt[-1], map_location="cpu", weights_only=False)
        net = SwiGLU_mlp(
            **model_args_for_state_dict(state["model_args"], state["model_state_dict"])
        )
        net.load_state_dict(state["model_state_dict"])
        net.eval()
        members.append((args.input_cols, net, tr))

    def predict(targets: pd.DataFrame) -> np.ndarray:
        """Mean over fold models, each with its own normalization."""
        preds = []
        for cols, net, tr in members:
            z = (raw_inputs(targets, cols) - tr.input_mean.double().numpy()) / (
                tr.input_std.double().numpy()
            )
            with torch.no_grad():
                out = net(torch.tensor(z, dtype=torch.float32)).double().numpy()
            preds.append(
                out * tr.target_std.double().numpy() + tr.target_mean.double().numpy()
            )
        return np.mean(preds, axis=0)

    return predict


def main() -> None:
    """Draw feasible targets, compute the four recipes, and write the CSVs."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n_targets", type=int, default=25, help="Target welds (x4 runs).")
    p.add_argument("--seed", type=int, default=20261120, help="Random seed.")
    p.add_argument("--outdir", default=".", help="Output directory.")
    opts = p.parse_args()
    rng = np.random.default_rng(opts.seed)

    data = pd.read_csv(TABLE)
    melted = data[data["melted"]]
    lo = {c: melted[c].quantile(0.1) for c in ("depth_um", "sigma_m")}
    hi = {c: melted[c].quantile(0.9) for c in ("depth_um", "sigma_m")}
    ls_lo, ls_hi = np.log10(melted["S_wpct"].min()), np.log10(melted["S_wpct"].max())
    rmin, rmax = melted[RECIPE].min().to_numpy(), melted[RECIPE].max().to_numpy()

    # Support check in z-scored (depth, sigma, log10 S).
    sup = np.c_[melted["depth_um"], melted["sigma_m"], np.log10(melted["S_wpct"])]
    sm, ss = sup.mean(0), sup.std(0)
    zs = (sup - sm) / ss
    dist = np.linalg.norm(zs[:, None] - zs[None], axis=2)
    np.fill_diagonal(dist, np.inf)
    nn_limit = np.quantile(dist.min(1), 0.9)

    recipes = {
        "MLP_withS": mlp_ensemble(
            HERE, "tpbf_inv_plan_kfold.csv", "train_tpbf_inverse_mlp.py"
        ),
        "quadratic_withS": quadratic_model(
            ["depth_um", "sigma_m", "S_wpct"], RECIPE, ["S_wpct"]
        ),
        "MLP_noS": mlp_ensemble(
            HERE, "tpbf_inv_plan_noS_kfold.csv", "train_tpbf_inverse_mlp.py"
        ),
        "quadratic_noS": quadratic_model(["depth_um", "sigma_m"], RECIPE, []),
    }
    fwd_cols = ["power_W", "scan_speed_mps", "sigma_m", "S_wpct"]
    forward = {
        "fwd_MLP": mlp_ensemble(FWD_DIR, "tpbf_kfold_fix.csv", "train_tpbf_mlp.py"),
        "fwd_quadratic": quadratic_model(fwd_cols, ["depth_um"], ["S_wpct"]),
    }

    kept, drawn, rejected = [], 0, {"support": 0, "recipe range": 0}
    while len(kept) < opts.n_targets:
        u = latin_hypercube(opts.n_targets, 3, rng)
        cand = pd.DataFrame(
            {
                "depth_um": lo["depth_um"] + u[:, 0] * (hi["depth_um"] - lo["depth_um"]),
                "sigma_m": lo["sigma_m"] + u[:, 1] * (hi["sigma_m"] - lo["sigma_m"]),
                "S_wpct": 10 ** (ls_lo + u[:, 2] * (ls_hi - ls_lo)),
            }
        )
        drawn += len(cand)
        zc = (
            np.c_[cand["depth_um"], cand["sigma_m"], np.log10(cand["S_wpct"])] - sm
        ) / ss
        near = np.linalg.norm(zc[:, None] - zs[None], axis=2).min(1) <= nn_limit
        rec = {k: f(cand) for k, f in recipes.items()}
        in_range = np.all(
            [np.all((r >= rmin) & (r <= rmax), axis=1) for r in rec.values()], axis=0
        )
        rejected["support"] += int((~near).sum())
        rejected["recipe range"] += int((near & ~in_range).sum())
        for i in np.flatnonzero(near & in_range):
            if len(kept) < opts.n_targets:
                kept.append((cand.iloc[i], {k: v[i] for k, v in rec.items()}))

    design, inputs = [], []
    for t, (target, recs) in enumerate(kept, 1):
        for source, (power, speed) in recs.items():
            run_id = f"val_t{t:02d}_{source}"
            row = {
                "run_id": run_id,
                "power_W": power,
                "scan_speed_mps": speed,
                "sigma_m": target["sigma_m"],
                "S_wpct": target["S_wpct"],
            }
            inputs.append(row)
            design.append(
                {
                    **row,
                    "target_id": t,
                    "target_depth_um": target["depth_um"],
                    "recipe_source": source,
                    "model_knows_S": source.endswith("withS"),
                }
            )
    inputs, design = pd.DataFrame(inputs), pd.DataFrame(design)
    for k, f in forward.items():
        design[f"expected_depth_{k}_um"] = f(design)[:, 0]
    os.makedirs(opts.outdir, exist_ok=True)
    inputs.to_csv(
        os.path.join(opts.outdir, "tpbf_validation_truchas_inputs.csv"),
        index=False,
        float_format="%.8g",
    )
    design.to_csv(
        os.path.join(opts.outdir, "tpbf_validation_design.csv"),
        index=False,
        float_format="%.8g",
    )
    print(f"targets kept {len(kept)} of {drawn} drawn; rejected: {rejected}")
    print(f"wrote {len(inputs)} runs to {opts.outdir}/")


if __name__ == "__main__":
    main()
