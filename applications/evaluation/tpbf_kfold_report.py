"""Build slide-ready figures and tables summarizing the TPBF k-fold MLP studies.

Collects the results of three rocinante harnesses (all on the same 10 k-fold
splits of the melted rows of ``tpbf_training_set_261105.csv``):

- ``applications/harnesses/ro_tpbf_mlp``: forward MLPs (process -> melt pool),
  studies 1-10 (v1), 21-30 (v2), and 31-40 (v3, with the input-LayerNorm fix);
- ``applications/harnesses/ro_tpbf_mlp_meas``: forward v3 MLP with the in-situ
  measurables ``max_T_K``/``umax_mps`` as extra inputs, studies 1-10;
- ``applications/harnesses/ro_tpbf_inverse_mlp``: inverse v3 MLPs, studies
  1-10 (depth, S, length, width -> power, speed, sigma), 11-20 (+ measurables),
  and 21-30 (known sigma -> power, speed).

It reads the ``summarize_kfold.py`` / ``baselines_kfold.py`` metric CSVs, the
prediction CSVs, and the per-epoch loss records under each harness's ``runs/``,
and writes to ``--outdir``:

- ``fig_*.png``: loss curves, accuracy comparisons, parity plots, and the
  inverse forward-consistency check (16:9-friendly, 200 dpi);
- ``table_*.csv`` and ``table_*.png``: the comparison tables;
- ``tables.html``: every table as an HTML table (open in a browser, copy a
  table, paste into PowerPoint to get a native, editable table), followed by
  the figures;
- ``tpbf_kfold_summary.pptx``: a 16:9 deck with one slide per table (native,
  editable PowerPoint tables) and one per figure (needs ``python-pptx``);
- ``README.md``: what each file shows, as slide captions.

Usage (from the repository root, with the Yoke environment)::

    python applications/evaluation/tpbf_kfold_report.py --outdir tpbf_report
"""

import argparse
import html
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from yoke.datasets.tabular_csv import TabularCSV_DataSet  # noqa: E402
from yoke.plots.loss_curves import find_record_csvs, load_record_csv  # noqa: E402

# --------------------------------------------------------------------------
# Style (dataviz reference palette, light mode; validated categorical order)
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Model identity is fixed across every figure (color follows the entity).
COLOR = {
    "mean": MUTED,
    "linear": "#2a78d6",
    "quadratic": "#eb6834",
    "MLP": "#1baf7a",
    "MLP v1": "#e87ba4",
    "MLP v2": "#eda100",
    "MLP v3": "#1baf7a",
}
TRAIN_C, HOLD_C = "#2a78d6", "#eb6834"
NO_MEAS_C = AXIS  # de-emphasized "before" bar in the measurables figure

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Display names and unit conversions (stored SI -> shown units).
LABEL = {
    "depth_um": ("Depth", "µm", 1.0),
    "width_um": ("Width", "µm", 1.0),
    "fwhm_um": ("FWHM", "µm", 1.0),
    "length_um": ("Length", "µm", 1.0),
    "power_W": ("Power", "W", 1.0),
    "scan_speed_mps": ("Scan speed", "mm/s", 1e3),
    "sigma_m": ("Beam sigma", "µm", 1e6),
}
FWD_T = ["depth_um", "width_um", "fwhm_um", "length_um"]
INV_T = ["power_W", "scan_speed_mps", "sigma_m"]

MLP_RUNS = {
    "MLP v1": "[64,64,64], wd 0.01, 1000 epochs (input-LayerNorm bug)",
    "MLP v2": "[32,32], wd 0.1, 150 epochs (input-LayerNorm bug)",
    "MLP v3": "[32,32], wd 0.1, 150 epochs (fixed model)",
}
INV_VARIANTS = {
    "inv": "Inverse: depth, S, length, width -> power, speed, sigma",
    "inv_meas": "Inverse + measurables (max T, max flow speed)",
    "inv_sigma": "Inverse with known sigma -> power, speed",
}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def read_metrics(path: str, model: str | None = None) -> pd.DataFrame:
    """Read a per-study metrics CSV, optionally relabeling its model.

    Args:
        path (str): ``summarize_kfold.py`` or ``baselines_kfold.py`` output.
        model (str | None): Model label to assign (summaries have none).

    Returns:
        pd.DataFrame: Rows of (model, study, target, metrics).
    """
    df = pd.read_csv(path)
    if model is not None:
        df["model"] = model
    return df


def baselines(path: str) -> pd.DataFrame:
    """Keep only the mean/linear/quadratic rows of a baselines metrics CSV.

    Args:
        path (str): A ``baselines_kfold.py`` metrics CSV.

    Returns:
        pd.DataFrame: The three baseline models.
    """
    df = read_metrics(path)
    return df[df["model"].isin(["mean", "linear", "quadratic"])]


def summarize(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean and std over studies of ``metric``, per model and target.

    Args:
        df (pd.DataFrame): Per-study metric rows.
        metric (str): Column to aggregate.

    Returns:
        pd.DataFrame: Columns ``model, target, mean, std``.
    """
    g = df.groupby(["model", "target"], sort=False)[metric]
    return g.agg(["mean", "std"]).reset_index()


def loss_curves(rundir: str, studies: list[int]) -> dict[str, pd.DataFrame]:
    """Per-epoch mean training/holdout loss for each study of one run.

    Args:
        rundir (str): A harness's ``runs`` directory.
        studies (list[int]): Study indices.

    Returns:
        dict[str, pd.DataFrame]: ``"training"``/``"validation"`` -> a frame
        indexed by epoch with one column per study.
    """
    out = {}
    for kind in ("training", "validation"):
        cols = {}
        for s in studies:
            paths, _ = find_record_csvs(rundir, s, kind)
            df = pd.concat([load_record_csv(p) for p in paths])
            cols[s] = df.groupby("Epoch")["Loss"].mean()
        out[kind] = pd.DataFrame(cols)
    return out


# --------------------------------------------------------------------------
# Forward-consistency check for the inverse models
# --------------------------------------------------------------------------
def _quad_features(x: np.ndarray) -> np.ndarray:
    """``[1, x, x_i x_j (i <= j)]`` features for a quadratic least-squares fit.

    Args:
        x (np.ndarray): ``(n, d)`` inputs.

    Returns:
        np.ndarray: ``(n, 1 + d + d(d+1)/2)`` design matrix.
    """
    d = x.shape[1]
    cols = [np.ones(len(x)), *x.T]
    cols += [x[:, i] * x[:, j] for i in range(d) for j in range(i, d)]
    return np.stack(cols, axis=1)


def forward_consistency(
    table_csv: str, inv_dir: str, variant: str, study_csv: str
) -> pd.DataFrame:
    """Melt pool produced by each inverse-predicted recipe vs the requested one.

    For each study (split), a quadratic forward model (power, speed, sigma,
    log10 S -> depth, width, length) is fitted on that split's training fold.
    It is applied to (a) the inverse MLP's predicted recipe for every
    held-out simulation (a known sigma is used as is), and (b) the true
    recipe, which gives the forward model's own error floor.

    Args:
        table_csv (str): The TPBF table.
        inv_dir (str): The inverse harness directory.
        variant (str): ``"inv"``, ``"inv_meas"``, or ``"inv_sigma"``.
        study_csv (str): The variant's study CSV.

    Returns:
        pd.DataFrame: Per (study, kind, target) median relative error.
    """
    fin = ["power_W", "scan_speed_mps", "sigma_m", "S_wpct"]
    ft = ["depth_um", "width_um", "length_um"]
    raw = pd.read_csv(table_csv).set_index("name")
    study = pd.read_csv(os.path.join(inv_dir, study_csv), index_col=0, comment="#")
    pred = pd.read_csv(os.path.join(inv_dir, f"kfold_{variant}_summary_predictions.csv"))
    rows = []
    for sid, g in pred.groupby("studyIDX"):
        tr = TabularCSV_DataSet(
            table_csv,
            fin,
            ft,
            split="train",
            fold=int(study.loc[sid, "FOLD"]),
            split_seed=int(study.loc[sid, "SPLIT_SEED"]),
            require_true=["melted"],
            stratify_col="stratum",
            log_cols=["S_wpct"],
        )
        xm, xs = tr.input_mean.double().numpy(), tr.input_std.double().numpy()
        ym, ys = tr.target_mean.double().numpy(), tr.target_std.double().numpy()
        w = np.linalg.lstsq(
            _quad_features(tr.inputs.double().numpy()),
            tr.targets.double().numpy(),
            rcond=None,
        )[0]
        r = raw.loc[g["name"]]
        sigma = g["sigma_m_pred"].to_numpy() if "sigma_m_pred" in g else r["sigma_m"]
        recipes = {
            "inverse-MLP recipe": np.c_[
                g["power_W_pred"], g["scan_speed_mps_pred"], sigma, r["S_wpct"]
            ],
            "true recipe (floor)": r[fin].to_numpy(dtype=float),
        }
        want = r[ft].to_numpy(dtype=float)
        for kind, x in recipes.items():
            x = x.astype(float).copy()
            x[:, 3] = np.log10(x[:, 3])
            got = (_quad_features((x - xm) / xs) @ w) * ys + ym
            rel = np.median(np.abs(got - want) / want, axis=0)
            for t, v in zip(ft, rel, strict=True):
                rows.append(
                    {
                        "variant": variant,
                        "kind": kind,
                        "studyIDX": sid,
                        "target": t,
                        "median_rel_err": v,
                    }
                )
    return pd.DataFrame(rows)


def planning_consistency(table_csv: str, fwd_dir: str, inv_dir: str) -> pd.DataFrame:
    """Depth produced by weld-planning recipes vs the requested depth.

    The planning inverse predicts (power, speed) from the desired depth, the
    known sigma, and (optionally) S wt%. For each split and held-out simulation,
    each recipe (the inverse MLP's, a quadratic least-squares inverse fitted on
    the same training fold, the training-mean recipe, and the true recipe) is
    run through two forward "judges" at the simulation's true sigma and S:
    a quadratic least-squares forward model and the trained forward MLP v3
    (``ro_tpbf_mlp`` studies 31-40), both fitted on the same training fold.

    Args:
        table_csv (str): The TPBF table.
        fwd_dir (str): The ``ro_tpbf_mlp`` harness directory.
        inv_dir (str): The ``ro_tpbf_inverse_mlp`` harness directory.

    Returns:
        pd.DataFrame: Per (variant, recipe, judge, study): ``median_rel``
        (median relative depth error) and ``within10`` (fraction within 10%).
    """
    import glob

    import torch

    from yoke.models.mlp import SwiGLU_mlp, model_args_for_state_dict

    fin = ["power_W", "scan_speed_mps", "sigma_m", "S_wpct"]
    ft = ["depth_um", "width_um", "fwhm_um", "length_um"]
    common = {"require_true": ["melted"], "stratify_col": "stratum"}
    raw = pd.read_csv(table_csv).set_index("name")
    fstudy = pd.read_csv(
        os.path.join(fwd_dir, "tpbf_kfold_fix.csv"), index_col=0, comment="#"
    )
    rows = []
    for variant, tag, ins, logs in (
        ("with S", "plan", ["depth_um", "sigma_m", "S_wpct"], ["S_wpct"]),
        ("without S", "plan_noS", ["depth_um", "sigma_m"], []),
    ):
        study = pd.read_csv(
            os.path.join(inv_dir, f"tpbf_inv_{tag}_kfold.csv"), index_col=0, comment="#"
        )
        pred = pd.read_csv(
            os.path.join(inv_dir, f"kfold_inv_{tag}_summary_predictions.csv")
        )
        for sid, g in pred.groupby("studyIDX"):
            fold, seed = int(study.loc[sid, "FOLD"]), int(study.loc[sid, "SPLIT_SEED"])
            split = {"fold": fold, "split_seed": seed, **common}
            # Judges, fitted on this split's training fold.
            tr = TabularCSV_DataSet(
                table_csv, fin, ft, split="train", log_cols=["S_wpct"], **split
            )
            xm, xs = tr.input_mean.double().numpy(), tr.input_std.double().numpy()
            ym, ys = tr.target_mean.double().numpy(), tr.target_std.double().numpy()
            w = np.linalg.lstsq(
                _quad_features(tr.inputs.double().numpy()),
                tr.targets.double().numpy(),
                rcond=None,
            )[0]
            fsid = int(
                fstudy[(fstudy.FOLD == fold) & (fstudy.SPLIT_SEED == seed)].index[0]
            )
            ck = torch.load(
                glob.glob(
                    os.path.join(
                        fwd_dir,
                        "runs",
                        f"study_{fsid:03d}",
                        "*_modelState_epoch0150.pth",
                    )
                )[0],
                map_location="cpu",
                weights_only=False,
            )
            net = SwiGLU_mlp(
                **model_args_for_state_dict(ck["model_args"], ck["model_state_dict"])
            )
            net.load_state_dict(ck["model_state_dict"])
            net.eval()

            def zin(
                x: np.ndarray, xm: np.ndarray = xm, xs: np.ndarray = xs
            ) -> np.ndarray:
                """Log10 S, then z-score with the judge's training-fold stats."""
                x = x.astype(float).copy()
                x[:, 3] = np.log10(x[:, 3])
                return (x - xm) / xs

            def quad_judge(
                x: np.ndarray,
                w: np.ndarray = w,
                ym: np.ndarray = ym,
                ys: np.ndarray = ys,
            ) -> np.ndarray:
                """Quadratic forward model's depth."""
                return ((_quad_features(zin(x)) @ w) * ys + ym)[:, 0]

            def mlp_judge(
                x: np.ndarray,
                net: torch.nn.Module = net,
                ym: np.ndarray = ym,
                ys: np.ndarray = ys,
            ) -> np.ndarray:
                """Forward MLP v3's depth."""
                with torch.no_grad():
                    z = torch.tensor(zin(x), dtype=torch.float32)
                    return (net(z).double().numpy() * ys + ym)[:, 0]

            # Recipes for the held-out simulations.
            itr = TabularCSV_DataSet(
                table_csv,
                ins,
                ["power_W", "scan_speed_mps"],
                split="train",
                log_cols=logs,
                **split,
            )
            iho = TabularCSV_DataSet(
                table_csv,
                ins,
                ["power_W", "scan_speed_mps"],
                split="holdout",
                log_cols=logs,
                **split,
            )
            if list(iho.row_names) != list(g["name"]):
                raise ValueError(f"study {sid}: holdout rows do not match")
            s_, m_ = iho.target_std.double().numpy(), iho.target_mean.double().numpy()
            wq = np.linalg.lstsq(
                _quad_features(itr.inputs.double().numpy()),
                itr.targets.double().numpy(),
                rcond=None,
            )[0]
            r = raw.loc[g["name"]]
            recipes = {
                "MLP v3": np.c_[g["power_W_pred"], g["scan_speed_mps_pred"]],
                "quadratic": (_quad_features(iho.inputs.double().numpy()) @ wq) * s_
                + m_,
                "mean": np.tile(m_, (len(g), 1)),
                "true recipe": r[["power_W", "scan_speed_mps"]].to_numpy(dtype=float),
            }
            want = r["depth_um"].to_numpy(dtype=float)
            for rk, pv in recipes.items():
                x = np.c_[pv, r["sigma_m"], r["S_wpct"]]
                for jk, judge in (("quadratic", quad_judge), ("MLP v3", mlp_judge)):
                    rel = np.abs(judge(x) - want) / want
                    rows.append(
                        {
                            "variant": variant,
                            "recipe": rk,
                            "judge": jk,
                            "studyIDX": sid,
                            "median_rel": np.median(rel),
                            "within10": np.mean(rel < 0.10),
                        }
                    )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def pct_table(
    summary: pd.DataFrame, models: list[tuple[str, str]], targets: list[str]
) -> pd.DataFrame:
    """Format ``mean ± std`` of a relative-error summary as percentages.

    Args:
        summary (pd.DataFrame): Output of :func:`summarize` (fractions).
        models (list[tuple[str, str]]): ``(model key, row label)`` in order.
        targets (list[str]): Target columns in order.

    Returns:
        pd.DataFrame: String cells, indexed by row label.
    """
    s = summary.set_index(["model", "target"])
    data = {
        LABEL[t][0]: [
            f"{100 * s.loc[(m, t), 'mean']:.1f} ± {100 * s.loc[(m, t), 'std']:.1f}"
            for m, _ in models
        ]
        for t in targets
    }
    return pd.DataFrame(data, index=[lab for _, lab in models])


def unit_table(
    summary: pd.DataFrame, models: list[tuple[str, str]], targets: list[str]
) -> pd.DataFrame:
    """Format ``mean ± std`` of an RMSE summary in display units.

    Args:
        summary (pd.DataFrame): Output of :func:`summarize` (SI units).
        models (list[tuple[str, str]]): ``(model key, row label)`` in order.
        targets (list[str]): Target columns in order.

    Returns:
        pd.DataFrame: String cells; column headers carry the unit.
    """
    s = summary.set_index(["model", "target"])
    data = {}
    for t in targets:
        name, unit, k = LABEL[t]
        data[f"{name} ({unit})"] = [
            f"{k * s.loc[(m, t), 'mean']:.3g} ± {k * s.loc[(m, t), 'std']:.2g}"
            for m, _ in models
        ]
    return pd.DataFrame(data, index=[lab for _, lab in models])


def _best_rows(table: pd.DataFrame) -> dict[str, int]:
    """Row position of the best value in each column of a formatted table.

    Lowest wins, except in "within ..." columns (a share of hits), where the
    highest wins. Reference rows labeled "True recipe ..." (a judge's floor)
    and non-numeric cells (e.g. 'input', '-') never win.

    Args:
        table (pd.DataFrame): Cells like ``"8.1 ± 3.4"``.

    Returns:
        dict[str, int]: Column -> row position of the best value.
    """

    def value(cell: str) -> float:
        """Leading number of a cell, or NaN when it is not numeric."""
        try:
            return float(cell.split("±")[0])
        except ValueError:
            return float("nan")

    skip = [str(i).startswith("True recipe") for i in table.index]
    best = {}
    for c in table.columns:
        higher = "within" in c.lower()
        vals = [
            float("nan") if s else value(v) for v, s in zip(table[c], skip, strict=True)
        ]
        vals = [(-v if higher else v) for v in vals]
        vals = [float("inf") if np.isnan(v) else v for v in vals]
        best[c] = int(np.argmin(vals))
    return best


def table_html(table: pd.DataFrame, title: str, note: str) -> str:
    """Render one table as simple, paste-friendly HTML (best value bold).

    Args:
        table (pd.DataFrame): Formatted cells.
        title (str): Caption shown above the table.
        note (str): Footnote shown below it.

    Returns:
        str: An HTML fragment.
    """
    best = _best_rows(table)
    td = 'style="border:1px solid #c3c2b7;padding:4px 10px;text-align:{a}"'
    head = "".join(f"<th {td.format(a='center')}>{html.escape(c)}</th>" for c in table)
    body = []
    for i, (label, row) in enumerate(table.iterrows()):
        cells = [f"<td {td.format(a='left')}>{html.escape(str(label))}</td>"]
        for c in table.columns:
            v = html.escape(row[c])
            v = f"<b>{v}</b>" if best[c] == i else v
            cells.append(f"<td {td.format(a='center')}>{v}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f"<h3>{html.escape(title)}</h3>\n"
        '<table style="border-collapse:collapse;font-family:Arial,sans-serif;'
        'font-size:11pt">\n'
        f"<tr><th {td.format(a='left')}>Model</th>{head}</tr>\n"
        + "\n".join(body)
        + f"\n</table>\n<p style='font-size:9pt;color:#52514e'>{html.escape(note)}</p>\n"
    )


def table_png(table: pd.DataFrame, title: str, note: str, path: str) -> None:
    """Render one table as a PNG image (best value per column in bold).

    The figure is sized to the text, so the image has no empty margins and can
    be dropped onto a slide as is.

    Args:
        table (pd.DataFrame): Formatted cells.
        title (str): Title drawn above the table.
        note (str): Footnote drawn below it (wrapped to the table width).
        path (str): Output PNG path.
    """
    cells = [
        [str(i), *row] for i, row in zip(table.index, table.to_numpy(), strict=True)
    ]
    header = ["Model", *table.columns]
    # Column widths from the longest string, in characters (about 0.085 in each
    # at 10 pt), plus padding.
    chars = [max(len(h), *(len(r[j]) for r in cells)) + 3 for j, h in enumerate(header)]
    width_in = 0.085 * sum(chars)
    nrow = len(cells) + 1
    wrapped = textwrap.fill(note, width=max(60, int(sum(chars) * 1.15)))
    n_note = wrapped.count("\n") + 1
    height_in = 0.3 * nrow + 0.45 + 0.17 * n_note
    fig = plt.figure(figsize=(width_in, height_in))
    table_h = 0.3 * nrow / height_in
    ax = fig.add_axes((0, (0.17 * n_note + 0.1) / height_in, 1, table_h))
    ax.axis("off")
    tab = ax.table(
        cellText=cells,
        colLabels=header,
        colWidths=[c / sum(chars) for c in chars],
        cellLoc="center",
        bbox=(0, 0, 1, 1),
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    best = _best_rows(table)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor(AXIS)
        cell.set_linewidth(0.6)
        cell.set_facecolor("#f0efec" if r == 0 else SURFACE)
        if r == 0:
            cell.set_text_props(color=INK, weight="bold")
        elif c == 0:
            cell.set_text_props(color=INK, ha="left")
            cell.PAD = 0.03
        elif best[table.columns[c - 1]] == r - 1:
            cell.set_text_props(weight="bold")
    fig.text(0.0, 1.0, title, fontsize=12, color=INK, ha="left", va="top")
    fig.text(0.0, 0.0, wrapped, fontsize=8.5, color=INK2, ha="left", va="bottom")
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def grouped_bars(
    ax: plt.Axes,
    summary: pd.DataFrame,
    models: list[tuple[str, str, str]],
    targets: list[str],
) -> None:
    """Draw grouped bars of mean median-relative-error (%) with std whiskers.

    Args:
        ax (plt.Axes): Target axes.
        summary (pd.DataFrame): Output of :func:`summarize`.
        models (list[tuple[str, str, str]]): ``(model key, legend label,
            color)`` in draw order.
        targets (list[str]): Targets along the x axis.
    """
    s = summary.set_index(["model", "target"])
    n = len(models)
    width = min(0.8 / n, 0.2)
    x = np.arange(len(targets))
    for i, (key, label, color) in enumerate(models):
        means = [100 * s.loc[(key, t), "mean"] for t in targets]
        stds = [100 * s.loc[(key, t), "std"] for t in targets]
        pos = x + (i - (n - 1) / 2) * width
        ax.bar(pos, means, width * 0.88, color=color, label=label, zorder=2)
        ax.errorbar(
            pos,
            means,
            yerr=stds,
            fmt="none",
            ecolor=INK2,
            elinewidth=1,
            capsize=2,
            zorder=3,
        )
    ax.set_xticks(x, [LABEL[t][0] for t in targets])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Holdout median relative error (%)")
    # Leave the top ~30% of the axes free for the legend.
    top = max(
        100 * (s.loc[(k, t), "mean"] + s.loc[(k, t), "std"])
        for k, _, _ in models
        for t in targets
    )
    ax.set_ylim(0, 1.4 * top)


def fig_loss_curves(
    panels: list[tuple[str, dict[str, pd.DataFrame]]], title: str, path: str
) -> None:
    """Small multiples of training vs holdout loss, median over the 10 splits.

    Args:
        panels (list[tuple[str, dict]]): ``(panel title, curves)`` pairs, where
            curves come from :func:`loss_curves`.
        title (str): Figure title.
        path (str): Output PNG path.
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.6), sharey=True, squeeze=False)
    for ax, (ptitle, curves) in zip(axes[0], panels, strict=True):
        for kind, color, label in (
            ("training", TRAIN_C, "Training folds"),
            ("validation", HOLD_C, "Holdout fold"),
        ):
            df = curves[kind]
            med = df.median(axis=1)
            ax.fill_between(
                df.index,
                df.min(axis=1),
                df.max(axis=1),
                color=color,
                alpha=0.12,
                linewidth=0,
            )
            ax.plot(
                med.index, med, color=color, lw=2, label=label, solid_capstyle="round"
            )
        ax.set_yscale("log")
        ax.set_title(ptitle, fontsize=10.5, loc="left")
        ax.set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss: L2 norm of z-scored error")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        ncols=2,
        fontsize=9.5,
        bbox_to_anchor=(0.995, 1.0),
    )
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK)
    fig.text(
        0.01,
        -0.02,
        "Lines: median over the 10 k-fold splits; bands: min-max.",
        fontsize=8.5,
        color=INK2,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_parity(
    rows: list[tuple[str, pd.DataFrame | None]],
    targets: list[str],
    title: str,
    path: str,
) -> None:
    """Predicted vs true on the held-out folds (both repeats; all simulations).

    Args:
        rows (list[tuple[str, pd.DataFrame | None]]): ``(row label,
            predictions)`` per figure row (``summarize_kfold.py`` output).
        targets (list[str]): One column of panels per target.
        title (str): Figure title.
        path (str): Output PNG path.
    """
    nr, nc = len(rows), len(targets)
    fig, axes = plt.subplots(nr, nc, figsize=(3.0 * nc, 2.9 * nr), squeeze=False)
    for i, (rlabel, pred) in enumerate(rows):
        for j, t in enumerate(targets):
            ax = axes[i, j]
            name, unit, k = LABEL[t]
            if f"{t}_pred" not in pred:
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    f"{name}: known input",
                    ha="center",
                    va="center",
                    color=INK2,
                    transform=ax.transAxes,
                )
                continue
            tru, prd = k * pred[f"{t}_true"], k * pred[f"{t}_pred"]
            lo, hi = min(tru.min(), prd.min()), max(tru.max(), prd.max())
            pad = 0.05 * (hi - lo)
            ax.plot(
                [lo - pad, hi + pad], [lo - pad, hi + pad], color=AXIS, lw=1, zorder=1
            )
            ax.scatter(
                tru,
                prd,
                s=16,
                color=COLOR["MLP v3"],
                alpha=0.75,
                edgecolors=SURFACE,
                linewidths=0.5,
                zorder=2,
            )
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=8.5)
            if i == 0:
                ax.set_title(f"{name} ({unit})", fontsize=10.5)
            if i == nr - 1:
                ax.set_xlabel("True", fontsize=9.5)
            if j == 0:
                ax.set_ylabel(f"{rlabel}\nPredicted", fontsize=9.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_pptx(
    path: str,
    figs: list[tuple[str, str]],
    tables: list[tuple[str, str, str, pd.DataFrame]],
    outdir: str,
) -> None:
    """Write a 16:9 slide deck: a title slide, one slide per table, one per figure.

    Tables are native PowerPoint tables (editable; best value per column in
    bold); figures are the PNGs, scaled to fit, with their caption below.
    Requires ``python-pptx``.

    Args:
        path (str): Output ``.pptx`` path.
        figs (list[tuple[str, str]]): ``(png file name, caption)`` pairs.
        tables (list[tuple[str, str, str, pd.DataFrame]]): ``(key, title, note,
            formatted table)`` tuples.
        outdir (str): Directory holding the PNGs.
    """
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.util import Emu, Inches, Pt

    ink, ink2 = RGBColor(0x0B, 0x0B, 0x0B), RGBColor(0x52, 0x51, 0x4E)
    head_fill = RGBColor(0xF0, 0xEF, 0xEC)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]
    margin = Inches(0.5)
    width = prs.slide_width - 2 * margin

    def text_box(
        slide: object,
        top: Emu,
        height: Emu,
        text: str,
        size: int,
        color: RGBColor,
        bold: bool = False,
    ) -> None:
        """Add a left-aligned, word-wrapped text box across the slide."""
        frame = slide.shapes.add_textbox(margin, top, width, height).text_frame
        frame.word_wrap = True
        run = frame.paragraphs[0].add_run()
        run.text = text
        run.font.size, run.font.bold, run.font.color.rgb = Pt(size), bold, color

    slide = prs.slides.add_slide(blank)
    text_box(slide, Inches(2.6), Inches(1.0), "TPBF melt-pool surrogates", 36, ink, True)
    text_box(
        slide,
        Inches(3.6),
        Inches(1.4),
        "k-fold cross-validated MLPs vs least-squares baselines: forward "
        "(laser process -> melt pool) and inverse (melt pool -> laser "
        "recipe), with and without in-situ measurables. 98 melted "
        "simulations, 10 splits (5 folds x 2 shuffles).",
        18,
        ink2,
    )

    for _, title, note, tab in tables:
        slide = prs.slides.add_slide(blank)
        text_box(slide, Inches(0.35), Inches(0.8), title, 24, ink, True)
        nrow, ncol = tab.shape[0] + 1, tab.shape[1] + 1
        # Fit the table between the title and a ~1 in footnote band.
        row_h = min(Inches(0.42), int((prs.slide_height - Inches(2.45)) / nrow))
        shape = slide.shapes.add_table(
            nrow, ncol, margin, Inches(1.35), width, row_h * nrow
        )
        grid = shape.table
        label_w = Inches(4.6) if ncol > 3 else Inches(6.0)
        grid.columns[0].width = label_w
        for j in range(1, ncol):
            grid.columns[j].width = int((width - label_w) / (ncol - 1))
        best = _best_rows(tab)
        header = ["Model", *tab.columns]
        for i in range(nrow):
            for j in range(ncol):
                cell = grid.cell(i, j)
                text = (
                    header[j]
                    if i == 0
                    else (str(tab.index[i - 1]) if j == 0 else tab.iloc[i - 1, j - 1])
                )
                cell.text = text
                para = cell.text_frame.paragraphs[0]
                para.font.size = Pt(14 if nrow <= 8 else 12)
                para.font.color.rgb = ink
                para.font.bold = i == 0 or (
                    j > 0 and i > 0 and best[tab.columns[j - 1]] == i - 1
                )
                cell.fill.solid()
                cell.fill.fore_color.rgb = (
                    head_fill if i == 0 else RGBColor(0xFF, 0xFF, 0xFF)
                )
        text_box(
            slide,
            Inches(1.45) + row_h * nrow,
            Inches(0.9),
            note + " Bold: best per column.",
            12,
            ink2,
        )

    for png, caption in figs:
        slide = prs.slides.add_slide(blank)
        pic_top, pic_max_h = Inches(0.4), Inches(5.9)
        pic = slide.shapes.add_picture(os.path.join(outdir, png), margin, pic_top)
        # Scale to fit the area above the caption, keeping the aspect ratio.
        scale = min(width / pic.width, pic_max_h / pic.height)
        pic.width, pic.height = int(pic.width * scale), int(pic.height * scale)
        pic.left = int((prs.slide_width - pic.width) / 2)
        text_box(
            slide, pic_top + pic.height + Inches(0.15), Inches(0.9), caption, 14, ink2
        )
    prs.save(path)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    """Load every result, then write the figures, tables, HTML, and README."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--harnesses",
        default="applications/harnesses",
        help="Directory holding the three ro_tpbf_* harnesses.",
    )
    p.add_argument("--outdir", default="tpbf_report", help="Output directory.")
    opts = p.parse_args()
    fwd = os.path.join(opts.harnesses, "ro_tpbf_mlp")
    meas = os.path.join(opts.harnesses, "ro_tpbf_mlp_meas")
    inv = os.path.join(opts.harnesses, "ro_tpbf_inverse_mlp")
    out = opts.outdir
    os.makedirs(out, exist_ok=True)
    figs, tables = [], []  # (file, caption)
    nsplit = "mean ± std over the 10 k-fold splits (5 folds x 2 shuffles)"
    bars = "mean over the 10 k-fold splits (5 folds x 2 shuffles)"

    # ---- Forward: baselines and MLP versions --------------------------------
    f_df = pd.concat(
        [
            baselines(os.path.join(fwd, "kfold_baselines_metrics.csv")),
            read_metrics(
                os.path.join(fwd, "kfold_summary_metrics_64c64c64_MLP.csv"), "MLP v1"
            ),
            read_metrics(os.path.join(fwd, "kfold_reg_summary_metrics.csv"), "MLP v2"),
            read_metrics(os.path.join(fwd, "kfold_fix_summary_metrics.csv"), "MLP v3"),
        ]
    )
    f_rel, f_rmse = summarize(f_df, "median_rel_err"), summarize(f_df, "rmse")
    f_rows = [
        ("mean", "Predict training mean"),
        ("linear", "Linear least squares"),
        ("quadratic", "Quadratic least squares"),
        ("MLP v1", "MLP v1: " + MLP_RUNS["MLP v1"]),
        ("MLP v2", "MLP v2: " + MLP_RUNS["MLP v2"]),
        ("MLP v3", "MLP v3: " + MLP_RUNS["MLP v3"]),
    ]
    tables.append(
        (
            "forward_rel_err",
            "Forward model: holdout median relative error (%)",
            f"Inputs: power, speed, beam sigma, log10 S wt%. {nsplit}. "
            "Bold: best per column.",
            pct_table(f_rel, f_rows, FWD_T),
        )
    )
    tables.append(
        (
            "forward_rmse",
            "Forward model: holdout RMSE",
            f"{nsplit}. Bold: best per column.",
            unit_table(f_rmse, f_rows, FWD_T),
        )
    )

    fig, ax = plt.subplots(figsize=(10, 4.6))
    grouped_bars(
        ax,
        f_rel,
        [
            (k, lab, COLOR[k])
            for k, lab in [
                ("mean", "Training mean"),
                ("linear", "Linear"),
                ("quadratic", "Quadratic"),
                ("MLP v3", "MLP v3"),
            ]
        ],
        FWD_T,
    )
    ax.legend(ncols=4, loc="upper right")
    ax.set_title(
        "Forward model (process -> melt pool): MLP vs simple baselines", loc="left"
    )
    fig.text(
        0.01,
        -0.03,
        f"Bars: {bars}; whiskers: ±1 std. Same splits for every model.",
        fontsize=8.5,
        color=INK2,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(out, "fig_forward_vs_baselines.png"), dpi=200, bbox_inches="tight"
    )
    plt.close(fig)
    figs.append(
        (
            "fig_forward_vs_baselines.png",
            "Forward model accuracy: the fixed MLP (v3) vs mean, linear, and "
            "quadratic least-squares baselines. The MLP is best for depth; the "
            "quadratic is best for width, FWHM, and length.",
        )
    )

    fig, ax = plt.subplots(figsize=(10, 4.6))
    grouped_bars(
        ax,
        f_rel,
        [(k, f"{k}: {MLP_RUNS[k]}", COLOR[k]) for k in MLP_RUNS]
        + [("quadratic", "Quadratic least squares", COLOR["quadratic"])],
        FWD_T,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        "Forward MLP: effect of regularization and the input-LayerNorm fix", loc="left"
    )
    fig.text(0.01, -0.03, f"Bars: {bars}; whiskers: ±1 std.", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(
        os.path.join(out, "fig_forward_mlp_versions.png"), dpi=200, bbox_inches="tight"
    )
    plt.close(fig)
    figs.append(
        (
            "fig_forward_mlp_versions.png",
            "Forward MLP versions on identical splits: regularization (v1 -> v2) "
            "helped modestly; removing the LayerNorm on the 4-feature input "
            "(v2 -> v3) halved the error.",
        )
    )

    # ---- Forward with measurables -------------------------------------------
    m_df = pd.concat(
        [
            baselines(os.path.join(meas, "kfold_baselines_metrics.csv")).assign(
                model=lambda d: d["model"] + " +meas"
            ),
            read_metrics(
                os.path.join(meas, "kfold_meas_summary_metrics.csv"), "MLP v3 +meas"
            ),
            f_df[f_df["model"].isin(["linear", "quadratic", "MLP v3"])],
        ]
    )
    m_rel, m_rmse = summarize(m_df, "median_rel_err"), summarize(m_df, "rmse")
    m_rows = [
        ("linear", "Linear, 4 process inputs"),
        ("linear +meas", "Linear + measurables"),
        ("quadratic", "Quadratic, 4 process inputs"),
        ("quadratic +meas", "Quadratic + measurables"),
        ("MLP v3", "MLP v3, 4 process inputs"),
        ("MLP v3 +meas", "MLP v3 + measurables"),
    ]
    meas_note = (
        "Measurables: max melt-pool surface temperature (max_T_K) and max "
        f"surface flow speed (umax_mps), from simulation. {nsplit}."
    )
    tables.append(
        (
            "measurables_rel_err",
            "Forward model with in-situ measurables: holdout median relative error (%)",
            meas_note + " Bold: best per column.",
            pct_table(m_rel, m_rows, FWD_T),
        )
    )
    tables.append(
        (
            "measurables_rmse",
            "Forward model with in-situ measurables: holdout RMSE",
            meas_note,
            unit_table(m_rmse, m_rows, FWD_T),
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, key, name in (
        (axes[0], "quadratic", "Quadratic least squares"),
        (axes[1], "MLP v3", "MLP v3"),
    ):
        grouped_bars(
            ax,
            m_rel,
            [
                (key, "4 process inputs", NO_MEAS_C),
                (f"{key} +meas", "+ max T, max flow speed", COLOR[key]),
            ],
            FWD_T,
        )
        ax.set_title(name, loc="left")
        ax.legend(loc="upper right", fontsize=9)
    axes[1].set_ylabel("")
    fig.suptitle(
        "Do in-situ measurables help? Forward model, same splits",
        x=0.01,
        ha="left",
        fontsize=13,
        color=INK,
    )
    fig.text(
        0.01,
        -0.03,
        f"Bars: {bars}; whiskers: ±1 std. Measurables are simulated (noise-free).",
        fontsize=8.5,
        color=INK2,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(out, "fig_measurables_effect.png"), dpi=200, bbox_inches="tight"
    )
    plt.close(fig)
    figs.append(
        (
            "fig_measurables_effect.png",
            "Adding max surface temperature and flow speed as inputs: depth "
            "error drops (RMSE about -25% for both models); width, FWHM, and "
            "length do not improve.",
        )
    )

    # ---- Inverse ------------------------------------------------------------
    inv_csv = {
        "inv": "tpbf_inv_kfold.csv",
        "inv_meas": "tpbf_inv_meas_kfold.csv",
        "inv_sigma": "tpbf_inv_sigma_kfold.csv",
    }
    i_parts = []
    for v in INV_VARIANTS:
        b = baselines(os.path.join(inv, f"kfold_{v}_baselines_metrics.csv"))
        mlp = read_metrics(os.path.join(inv, f"kfold_{v}_summary_metrics.csv"), "MLP")
        i_parts.append(pd.concat([b, mlp]).assign(variant=v))
    i_df = pd.concat(i_parts)
    i_df["model"] = i_df["variant"] + "|" + i_df["model"]
    i_rel, i_rmse = summarize(i_df, "median_rel_err"), summarize(i_df, "rmse")
    vshort = {
        "inv": "No measurables",
        "inv_meas": "+ measurables",
        "inv_sigma": "Known sigma",
    }
    i_rows_all = [
        (f"{v}|{m}", f"{vshort[v]}: {lab}")
        for v in INV_VARIANTS
        for m, lab in (
            ("mean", "training mean"),
            ("linear", "linear"),
            ("quadratic", "quadratic"),
            ("MLP", "MLP v3"),
        )
    ]
    # The known-sigma variant has no sigma target; show "input" there.
    rel_tab = pct_table(
        i_rel[~i_rel["target"].eq("sigma_m")], i_rows_all, ["power_W", "scan_speed_mps"]
    )
    sig = pct_table(
        i_rel[i_rel["model"].str.startswith(("inv|", "inv_meas|"))],
        [r for r in i_rows_all if not r[0].startswith("inv_sigma")],
        ["sigma_m"],
    )
    rel_tab[LABEL["sigma_m"][0]] = (
        sig[LABEL["sigma_m"][0]].reindex(rel_tab.index).fillna("input")
    )
    tables.append(
        (
            "inverse_rel_err",
            "Inverse model (melt pool -> laser recipe): holdout median "
            "relative error of the predicted parameters (%)",
            "Inputs: requested depth, observed S wt%, length, width (+ max T "
            "and max flow speed, or a known sigma). 'input': sigma is "
            f"given, not predicted. {nsplit}.",
            rel_tab,
        )
    )
    # Same layout in physical units (sigma again only for the first two).
    rmse_tab = unit_table(
        i_rmse[~i_rmse["target"].eq("sigma_m")],
        i_rows_all,
        ["power_W", "scan_speed_mps"],
    )
    sig_col = "{} ({})".format(*LABEL["sigma_m"][:2])
    sig_rmse = unit_table(
        i_rmse[i_rmse["model"].str.startswith(("inv|", "inv_meas|"))],
        [r for r in i_rows_all if not r[0].startswith("inv_sigma")],
        ["sigma_m"],
    )
    rmse_tab[sig_col] = sig_rmse[sig_col].reindex(rmse_tab.index).fillna("input")
    tables.append(
        (
            "inverse_rmse",
            "Inverse model: holdout RMSE of the predicted parameters",
            f"'input': sigma is given, not predicted. {nsplit}.",
            rmse_tab,
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    for ax, v in zip(axes, INV_VARIANTS, strict=True):
        tgts = INV_T if v != "inv_sigma" else INV_T[:2]
        grouped_bars(
            ax,
            i_rel,
            [
                (f"{v}|{k}", lab, COLOR[k])
                for k, lab in [
                    ("mean", "Training mean"),
                    ("linear", "Linear"),
                    ("quadratic", "Quadratic"),
                    ("MLP", "MLP v3"),
                ]
            ],
            tgts,
        )
        ax.set_title(vshort[v], loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        ncols=4,
        fontsize=9.5,
        bbox_to_anchor=(0.995, 1.0),
    )
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.suptitle(
        "Inverse model: error of the predicted laser parameters",
        x=0.01,
        ha="left",
        fontsize=13,
        color=INK,
    )
    fig.text(
        0.01,
        -0.03,
        f"Bars: {bars}; whiskers: ±1 std. Without measurables, "
        "sigma is no better than predicting the mean.",
        fontsize=8.5,
        color=INK2,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(out, "fig_inverse_vs_baselines.png"), dpi=200, bbox_inches="tight"
    )
    plt.close(fig)
    figs.append(
        (
            "fig_inverse_vs_baselines.png",
            "Inverse accuracy for the three variants. Power is well determined "
            "(3-6%); speed is hardest (12-19%); sigma is only recoverable with "
            "the measurables. The quadratic beats the MLP throughout.",
        )
    )

    # ---- Inverse forward-consistency -----------------------------------------
    table_csv = os.path.join(fwd, "tpbf_training_set_261105.csv")
    fc = pd.concat(
        [forward_consistency(table_csv, inv, v, inv_csv[v]) for v in INV_VARIANTS]
    )
    fc["model"] = fc["variant"] + "|" + fc["kind"]
    fc_sum = summarize(fc, "median_rel_err")
    fc_targets = ["depth_um", "width_um", "length_um"]
    fc_rows = [
        (f"{v}|inverse-MLP recipe", f"{vshort[v]}: MLP recipe") for v in INV_VARIANTS
    ] + [("inv|true recipe (floor)", "True recipe (forward-model floor)")]
    tables.append(
        (
            "inverse_consistency",
            "Inverse recipes: does the predicted recipe give the requested "
            "melt pool? Median relative error (%)",
            "Each held-out recipe from the inverse MLP is run through a "
            "quadratic forward model fitted on the same training fold; "
            "'floor' feeds that forward model the true recipe. A surrogate "
            f"check, not a simulation. {nsplit}.",
            pct_table(fc_sum, fc_rows, fc_targets),
        )
    )
    fig, ax = plt.subplots(figsize=(10, 4.4))
    fc_models = [
        (f"{v}|inverse-MLP recipe", f"MLP recipe, {vshort[v].lower()}", c)
        for v, c in zip(INV_VARIANTS, ("#2a78d6", "#eb6834", "#1baf7a"), strict=True)
    ]
    fc_models.append(
        ("inv|true recipe (floor)", "True recipe (forward-model floor)", MUTED)
    )
    grouped_bars(ax, fc_sum, fc_models, fc_targets)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        "Inverse recipes reach the requested melt pool (surrogate check)", loc="left"
    )
    fig.text(
        0.01,
        -0.03,
        "Error of the melt pool a recipe produces vs the one "
        "requested, judged by a quadratic forward model per split.",
        fontsize=8.5,
        color=INK2,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(out, "fig_inverse_consistency.png"), dpi=200, bbox_inches="tight"
    )
    plt.close(fig)
    figs.append(
        (
            "fig_inverse_consistency.png",
            "Forward-consistency of the inverse MLP's recipes: errors on the "
            "parameters themselves (e.g. sigma) matter little; the produced "
            "width/length are within about 4-5% of the request.",
        )
    )

    # ---- In-situ S wt% utility: every S-free study vs its with-S twin -------
    s_note = (
        "Each S-free study is the exact twin of its with-S study: same splits "
        f"(same held-out simulations), model, and training; only S wt% is "
        f"removed from the inputs. {nsplit}."
    )
    fs = pd.concat(
        [
            f_df[f_df["model"].isin(["quadratic", "MLP v3"])].assign(
                model=lambda d: d["model"] + "|S"
            ),
            baselines(os.path.join(fwd, "kfold_noS_baselines_metrics.csv"))
            .query("model == 'quadratic'")
            .assign(model="quadratic|noS"),
            read_metrics(
                os.path.join(fwd, "kfold_noS_summary_metrics.csv"), "MLP v3|noS"
            ),
            m_df[m_df["model"].isin(["quadratic +meas", "MLP v3 +meas"])].assign(
                model=lambda d: d["model"] + "|S"
            ),
            baselines(os.path.join(meas, "kfold_meas_noS_baselines_metrics.csv"))
            .query("model == 'quadratic'")
            .assign(model="quadratic +meas|noS"),
            read_metrics(
                os.path.join(meas, "kfold_meas_noS_summary_metrics.csv"),
                "MLP v3 +meas|noS",
            ),
        ]
    )
    fs_rel = summarize(fs, "median_rel_err")
    fs_rows = [
        (f"{m}|{c}", f"{lab}, {'with' if c == 'S' else 'without'} S wt%")
        for m, lab in (
            ("quadratic", "Quadratic"),
            ("MLP v3", "MLP v3"),
            ("quadratic +meas", "Quadratic + measurables"),
            ("MLP v3 +meas", "MLP v3 + measurables"),
        )
        for c in ("S", "noS")
    ]
    tables.append(
        (
            "S_utility_forward",
            "Value of in-situ S wt%: forward model (process -> melt pool), "
            "holdout median relative error (%)",
            s_note,
            pct_table(fs_rel, fs_rows, FWD_T),
        )
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharey="row")
    for i, (suffix, row_name) in enumerate(
        (("", "power, speed, sigma"), (" +meas", "+ max T, max flow speed"))
    ):
        for j, key in enumerate(("quadratic", "MLP v3")):
            ax = axes[i, j]
            grouped_bars(
                ax,
                fs_rel,
                [
                    (f"{key}{suffix}|S", "with S wt%", COLOR[key]),
                    (f"{key}{suffix}|noS", "without S wt%", NO_MEAS_C),
                ],
                FWD_T,
            )
            name = "Quadratic least squares" if key == "quadratic" else "MLP v3"
            ax.set_title(f"{name}; inputs: {row_name}", loc="left", fontsize=11)
            ax.legend(loc="upper right", fontsize=9)
            if j:
                ax.set_ylabel("")
    fig.suptitle(
        "Value of an in-situ S wt% measurement for predicting the melt pool",
        x=0.01,
        ha="left",
        fontsize=13,
        color=INK,
    )
    fig.text(
        0.01,
        -0.02,
        f"Bars: {bars}; whiskers: ±1 std. The two rows have different y scales. "
        f"{s_note.split(':')[0]}.",
        fontsize=8.5,
        color=INK2,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(out, "fig_S_utility_forward.png"), dpi=200, bbox_inches="tight"
    )
    plt.close(fig)
    figs.append(
        (
            "fig_S_utility_forward.png",
            "Without S wt%, depth error from process parameters alone roughly "
            "triples (about 8-10% -> 27-28%) and FWHM doubles. With the in-situ "
            "measurables as inputs, dropping S costs little: they carry most of "
            "its information.",
        )
    )

    # Inverse: parameter error and recipe consistency, with vs without S.
    nos_csv = {
        "inv_noS": "tpbf_inv_noS_kfold.csv",
        "inv_meas_noS": "tpbf_inv_meas_noS_kfold.csv",
        "inv_sigma_noS": "tpbf_inv_sigma_noS_kfold.csv",
    }
    fc_nos = pd.concat(
        [forward_consistency(table_csv, inv, v, c) for v, c in nos_csv.items()]
    )
    fc_all = pd.concat([fc, fc_nos])
    fc_all = fc_all[fc_all["kind"] == "inverse-MLP recipe"].assign(
        model=lambda d: d["variant"]
    )
    fc_all_sum = summarize(fc_all, "median_rel_err")
    ip = []
    for v in INV_VARIANTS:
        for c, suffix in (("S", ""), ("noS", "_noS")):
            mlp = read_metrics(
                os.path.join(inv, f"kfold_{v}{suffix}_summary_metrics.csv"), "MLP"
            )
            ip.append(mlp.assign(model=f"{v}|{c}"))
    ip_sum = summarize(pd.concat(ip), "median_rel_err").set_index(["model", "target"])
    cs = fc_all_sum.set_index(["model", "target"])
    rows, labels = [], []
    for v in INV_VARIANTS:
        for c, suffix in (("S", ""), ("noS", "_noS")):
            cell = {}
            for t in INV_T:
                if (f"{v}|{c}", t) in ip_sum.index:
                    m_, s_ = ip_sum.loc[(f"{v}|{c}", t), ["mean", "std"]]
                    cell[LABEL[t][0]] = f"{100 * m_:.1f} ± {100 * s_:.1f}"
                else:
                    cell[LABEL[t][0]] = "input"
            for t in ("depth_um", "width_um", "length_um"):
                m_, s_ = cs.loc[(f"{v}{suffix}", t), ["mean", "std"]]
                cell[f"Recipe -> {LABEL[t][0].lower()}"] = (
                    f"{100 * m_:.1f} ± {100 * s_:.1f}"
                )
            rows.append(cell)
            labels.append(f"{vshort[v]}, {'with' if c == 'S' else 'without'} S wt%")
    tables.append(
        (
            "S_utility_inverse",
            "Value of in-situ S wt%: inverse model (MLP v3), holdout median "
            "relative error (%)",
            "Left: error of the predicted parameters. 'Recipe ->': error of the "
            "melt pool the predicted recipe produces at the material's true S wt% "
            "vs the requested one, judged by a quadratic forward model per split "
            "(its own floor: depth 10.1, width 2.2, length 2.2). " + s_note,
            pd.DataFrame(rows, index=labels),
        )
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    for ax, v in zip(axes, INV_VARIANTS, strict=True):
        grouped_bars(
            ax,
            fc_all_sum,
            [
                (v, "MLP recipe, with S wt%", COLOR["MLP v3"]),
                (f"{v}_noS", "MLP recipe, without S wt%", NO_MEAS_C),
            ],
            ["depth_um", "width_um", "length_um"],
        )
        ax.set_title(vshort[v], loc="left")
    for ax in axes[1:]:
        ax.set_ylabel("")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_,
        loc="upper right",
        ncols=2,
        fontsize=9.5,
        bbox_to_anchor=(0.995, 1.0),
    )
    fig.suptitle(
        "Inverse recipes with and without S wt%: melt pool produced vs requested",
        x=0.01,
        ha="left",
        fontsize=13,
        color=INK,
    )
    fig.text(
        0.01,
        -0.03,
        "Judged at each simulation's true S wt% by a quadratic forward model per "
        "split, whose own depth error is about 10%: depth differences below that "
        "are not resolved.",
        fontsize=8.5,
        color=INK2,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(out, "fig_S_utility_inverse.png"), dpi=200, bbox_inches="tight"
    )
    plt.close(fig)
    figs.append(
        (
            "fig_S_utility_inverse.png",
            "Inverse recipes barely change without S wt%: the observed melt-pool "
            "length and width already carry S's effect. The forward-model judge "
            "(depth floor about 10%) cannot resolve the small depth differences; a "
            "fair test needs simulations or a better forward model.",
        )
    )

    # ---- Weld planning: desired depth + sigma (+ S) -> power, speed -----------
    pc = planning_consistency(table_csv, fwd, inv)
    pc_sum = pc.groupby(["variant", "recipe", "judge"], sort=False)[
        ["median_rel", "within10"]
    ].agg(["mean", "std"])
    plan_par = {}
    for variant, tag in (("with S", "plan"), ("without S", "plan_noS")):
        b = baselines(os.path.join(inv, f"kfold_inv_{tag}_baselines_metrics.csv"))
        mlp = read_metrics(
            os.path.join(inv, f"kfold_inv_{tag}_summary_metrics.csv"), "MLP v3"
        )
        plan_par[variant] = summarize(pd.concat([b, mlp]), "median_rel_err").set_index(
            ["model", "target"]
        )

    def mstd(mean: float, std: float) -> str:
        """Format a fraction's mean ± std as percentages."""
        return f"{100 * mean:.1f} ± {100 * std:.1f}"

    prow, plab = [], []
    for recipe in ("mean", "quadratic", "MLP v3", "true recipe"):
        for variant in ("with S", "without S"):
            cell = {}
            for t in ("power_W", "scan_speed_mps"):
                if recipe == "true recipe":
                    cell[f"{LABEL[t][0]} error"] = "-"
                else:
                    v = plan_par[variant].loc[(recipe, t)]
                    cell[f"{LABEL[t][0]} error"] = mstd(v["mean"], v["std"])
            for judge in ("MLP v3", "quadratic"):
                v = pc_sum.loc[(variant, recipe, judge)]
                cell[f"Depth error ({judge} judge)"] = mstd(
                    v[("median_rel", "mean")], v[("median_rel", "std")]
                )
            v = pc_sum.loc[(variant, recipe, "MLP v3")]
            cell["Welds within 10% of depth"] = mstd(
                v[("within10", "mean")], v[("within10", "std")]
            )
            prow.append(cell)
            name = {
                "mean": "Training-mean recipe",
                "quadratic": "Quadratic inverse",
                "MLP v3": "MLP v3 inverse",
                "true recipe": "True recipe (judge floor)",
            }[recipe]
            plab.append(f"{name}, {'with' if variant == 'with S' else 'without'} S wt%")
    plan_tab = pd.DataFrame(prow, index=plab)
    tables.append(
        (
            "weld_planning",
            "Weld planning: desired depth + beam sigma (+ S wt%) -> power, scan "
            "speed. Errors in %",
            "Parameter errors: holdout median relative error of the predicted "
            "power/speed (many recipes give the same depth, so these are not the "
            "goal). Depth error: median relative error of the depth the recipe "
            "produces at the true S wt% vs the requested depth, judged by the "
            "forward MLP v3 or a quadratic forward model fitted on the same "
            "training fold. 'Within 10%': share of held-out welds (MLP judge). "
            f"Floor rows are excluded from the bold best-per-column. {nsplit}.",
            plan_tab,
        )
    )
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(3)
    width = 0.32
    for k, (variant, color) in enumerate(
        (("with S", COLOR["MLP v3"]), ("without S", NO_MEAS_C))
    ):
        means = [
            100 * pc_sum.loc[(variant, r, "MLP v3"), ("median_rel", "mean")]
            for r in ("mean", "quadratic", "MLP v3")
        ]
        stds = [
            100 * pc_sum.loc[(variant, r, "MLP v3"), ("median_rel", "std")]
            for r in ("mean", "quadratic", "MLP v3")
        ]
        pos = x + (k - 0.5) * width
        ax.bar(
            pos,
            means,
            width * 0.88,
            color=color,
            label=f"{'With' if variant == 'with S' else 'Without'} S wt%",
            zorder=2,
        )
        ax.errorbar(
            pos,
            means,
            yerr=stds,
            fmt="none",
            ecolor=INK2,
            elinewidth=1,
            capsize=2,
            zorder=3,
        )
    floor = 100 * pc_sum.loc[("with S", "true recipe", "MLP v3"), ("median_rel", "mean")]
    ax.axhline(floor, color=INK2, lw=1.2, zorder=1)
    ax.text(
        2.55,
        floor,
        f"  judge floor\n  (true recipe, {floor:.1f}%)",
        va="center",
        fontsize=9,
        color=INK2,
    )
    ax.set_xticks(x, ["Training-mean recipe", "Quadratic inverse", "MLP v3 inverse"])
    ax.set_xlim(-0.6, 3.1)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(
        0,
        1.4
        * max(100 * (pc_sum[("median_rel", "mean")] + pc_sum[("median_rel", "std")])),
    )
    ax.set_ylabel("Depth error of the recipe (%)")
    ax.legend(loc="upper left")
    ax.set_title(
        "Weld planning: does the recipe deliver the requested depth?", loc="left"
    )
    fig.text(
        0.01,
        -0.03,
        textwrap.fill(
            "Inputs: desired depth, beam sigma, +/- S wt%. Depth the recipe "
            "produces at the true S wt%, judged by the forward MLP v3 for each "
            f"split. Bars: {bars}; whiskers: ±1 std.",
            width=150,
        ),
        fontsize=8.5,
        color=INK2,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_weld_planning.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    figs.append(
        (
            "fig_weld_planning.png",
            "Weld planning from desired depth and beam sigma: with a measured S "
            "wt%, recipes deliver the requested depth to about 7% (at the judge's "
            "own floor); without S, the MLP recipe is no better than a fixed "
            "average recipe (about 15%). With-S recipes win in 10 of 10 splits "
            "under both judges.",
        )
    )

    # ---- Loss curves -------------------------------------------------------
    fig_loss_curves(
        [
            (
                "MLP v1 (64x3, 1000 ep)",
                loss_curves(os.path.join(fwd, "runs"), list(range(1, 11))),
            ),
            (
                "MLP v2 (32x2 reg.)",
                loss_curves(os.path.join(fwd, "runs"), list(range(21, 31))),
            ),
            (
                "MLP v3 (fixed)",
                loss_curves(os.path.join(fwd, "runs"), list(range(31, 41))),
            ),
            (
                "MLP v3 + measurables",
                loss_curves(os.path.join(meas, "runs"), list(range(1, 11))),
            ),
        ],
        "Forward MLPs: training vs holdout loss",
        os.path.join(out, "fig_loss_forward.png"),
    )
    figs.append(
        (
            "fig_loss_forward.png",
            "Forward MLP training curves. v1 memorizes its training folds (train "
            "loss -> 0.02) while its holdout loss stays near 1.0; v2 barely fits; "
            "the fixed v3 reaches about half of v2's holdout loss.",
        )
    )
    fig_loss_curves(
        [
            (vshort[v], loss_curves(os.path.join(inv, "runs"), list(range(a, a + 10))))
            for v, a in (("inv", 1), ("inv_meas", 11), ("inv_sigma", 21))
        ],
        "Inverse MLPs: training vs holdout loss",
        os.path.join(out, "fig_loss_inverse.png"),
    )
    figs.append(
        (
            "fig_loss_inverse.png",
            "Inverse MLP training curves (loss over the z-scored target "
            "vector; the known-sigma variant has 2 targets, the others 3).",
        )
    )

    # ---- Parity plots --------------------------------------------------------
    fig_parity(
        [
            (
                "MLP v3",
                pd.read_csv(os.path.join(fwd, "kfold_fix_summary_predictions.csv")),
            ),
            (
                "MLP v3 + meas.",
                pd.read_csv(os.path.join(meas, "kfold_meas_summary_predictions.csv")),
            ),
        ],
        FWD_T,
        "Forward MLP: predicted vs true on held-out simulations",
        os.path.join(out, "fig_parity_forward.png"),
    )
    figs.append(
        (
            "fig_parity_forward.png",
            "Forward parity plots, every melted simulation held out twice (2 "
            "shuffles). The outliers at the low end are the two barely-melted "
            "pools (p2-092, p2-100).",
        )
    )
    fig_parity(
        [
            (
                vshort[v],
                pd.read_csv(os.path.join(inv, f"kfold_{v}_summary_predictions.csv")),
            )
            for v in INV_VARIANTS
        ],
        INV_T,
        "Inverse MLP: predicted vs true laser parameters (held out)",
        os.path.join(out, "fig_parity_inverse.png"),
    )
    figs.append(
        (
            "fig_parity_inverse.png",
            "Inverse parity plots. Without measurables, predicted sigma "
            "collapses toward the mean (flat cloud); with them it tracks the "
            "truth.",
        )
    )

    # ---- Write tables (CSV, PNG, HTML) and the README -------------------------
    html_parts = [
        "<html><head><meta charset='utf-8'><title>TPBF k-fold summary"
        "</title></head><body style='font-family:Arial,sans-serif'>",
        "<h2>TPBF melt-pool surrogates: k-fold summary tables</h2>",
        "<p>Copy a table from this page and paste it into PowerPoint to "
        "get a native, editable table.</p>",
    ]
    for key, title, note, tab in tables:
        tab.to_csv(
            os.path.join(out, f"table_{key}.csv"),
            index_label="Model",
            encoding="utf-8-sig",  # BOM so Excel reads the ± signs correctly
        )
        table_png(tab, title, note, os.path.join(out, f"table_{key}.png"))
        html_parts.append(table_html(tab, title, note))
    html_parts.append("<h2>Figures</h2>")
    for f, cap in figs:
        html_parts.append(
            f"<p><img src='{f}' style='max-width:1100px'><br>{html.escape(cap)}</p>"
        )
    html_parts.append("</body></html>")
    with open(os.path.join(out, "tables.html"), "w") as fh:
        fh.write("\n".join(html_parts))
    with open(os.path.join(out, "README.md"), "w") as fh:
        fh.write(
            "# TPBF k-fold summary: figures and tables\n\n"
            "Generated by `applications/evaluation/tpbf_kfold_report.py`.\n"
            "Every result uses the same 10 k-fold splits (5 folds x 2 "
            "shuffles) of the 98 melted simulations.\n\n## Tables\n\n"
        )
        for key, title, note, _ in tables:
            fh.write(f"- `table_{key}.csv|png`: {title}. {note}\n")
        fh.write(
            "\n`tables.html` holds every table (copy and paste into "
            "PowerPoint).\n\n## Figures\n\n"
        )
        for f, cap in figs:
            fh.write(f"- `{f}`: {cap}\n")
    write_pptx(os.path.join(out, "tpbf_kfold_summary.pptx"), figs, tables, out)
    print(
        f"Wrote {len(figs)} figures and {len(tables)} tables to {out}/ "
        "(and tpbf_kfold_summary.pptx)"
    )


if __name__ == "__main__":
    main()
