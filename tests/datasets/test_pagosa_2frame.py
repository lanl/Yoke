"""Tests for the canonical-channel 2-frame PAGOSA datasets.

All data is synthetic and written to ``tmp_path``; no real simulation files
are needed.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from yoke.datasets import pagosa_2frame as p2
from yoke.datasets.pagosa_fields import CANONICAL_FIELDS, DATASET_IDS

H, W = 4, 3


class _FixedRNG:
    """Stand-in RNG returning queued integers (gap_in, gap_out, prev, ...)."""

    def __init__(self, values: list[int]) -> None:
        """Store the queue."""
        self.values = list(values)

    def integers(self, low: int, high: int, endpoint: bool = False) -> int:
        """Pop the next queued integer."""
        return self.values.pop(0)


def _frame(path: Path, sim_time: float | None, **fields: np.ndarray) -> None:
    """Write one synthetic NPZ frame."""
    if sim_time is not None:
        fields["sim_time"] = np.array(sim_time)
    np.savez(path, **fields)


def _const(value: float) -> np.ndarray:
    """Return an ``(H, W)`` constant image."""
    return np.full((H, W), value, dtype=float)


def _lsc_fields(scale: float) -> dict[str, np.ndarray]:
    """Return LSC-style part-named fields; NaN marks outside the material."""
    fields = {"Uvelocity": _const(scale), "Wvelocity": _const(-scale)}
    for i, part in enumerate(["case", "throw", "maincharge"]):
        vofm = np.zeros((H, W))
        vofm[i] = 1.0
        dens = np.where(vofm > 0, scale * (i + 1), np.nan)
        fields[f"vofm_{part}"] = vofm
        fields[f"density_{part}"] = dens
        fields[f"pressure_{part}"] = dens / 10
        fields[f"energy_{part}"] = dens / 100
    return fields


def _prefix_file(tmp_path: Path, *prefixes: str) -> str:
    """Write a prefix list and return its path."""
    path = tmp_path / "prefixes.txt"
    path.write_text("\n".join(prefixes) + "\n")
    return str(path)


def _write_lsc_triplet(tmp_path: Path, times: tuple) -> None:
    """Write LSC frames idx 1-3 for prefix lsc240420_id00001."""
    for idx, t in zip((1, 2, 3), times):
        _frame(
            tmp_path / f"lsc240420_id00001_pvi_idx{idx:05d}.npz",
            t,
            **_lsc_fields(float(idx)),
        )


# --------------------------------------------------------------------------
# merge_material_parts
# --------------------------------------------------------------------------
def test_merge_single_part_passthrough() -> None:
    """One part: density is vf-weighted, other kinds are unchanged."""
    v, f = np.array([2.0, 4.0]), np.array([0.5, 1.0])
    assert np.allclose(p2.merge_material_parts("density", [v], [f]), [1.0, 4.0])
    assert np.array_equal(p2.merge_material_parts("pressure", [v], [f]), v)


def test_merge_mixed_cell_rules() -> None:
    """Density sums, pressure is vf-averaged, energy is mass-averaged."""
    rho = [np.array([1.0, 0.0]), np.array([3.0, 0.0])]
    vf = [np.array([0.25, 0.0]), np.array([0.75, 0.0])]
    p = [np.array([2.0, 0.0]), np.array([6.0, 0.0])]
    e = [np.array([10.0, 0.0]), np.array([20.0, 0.0])]

    dens = p2.merge_material_parts("density", rho, vf)
    pres = p2.merge_material_parts("pressure", p, vf)
    ener = p2.merge_material_parts("energy", e, vf, densities=rho)

    assert np.allclose(dens, [0.25 + 2.25, 0.0])
    assert np.allclose(pres, [(0.5 + 4.5) / 1.0, 0.0])
    # mass weights 0.25 and 2.25
    assert np.allclose(ener, [(2.5 + 45.0) / 2.5, 0.0])


def test_merge_disjoint_parts_equals_fill() -> None:
    """Non-overlapping parts merge to the per-cell value of the occupying part."""
    vf = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    p = [np.array([2.0, 0.0]), np.array([0.0, 5.0])]
    assert np.allclose(p2.merge_material_parts("pressure", p, vf), [2.0, 5.0])


def test_merge_bad_inputs() -> None:
    """Missing densities, unknown kinds, and length mismatches raise."""
    v = [np.ones(2), np.ones(2)]
    with pytest.raises(ValueError, match="density per part"):
        p2.merge_material_parts("energy", v, v)
    with pytest.raises(ValueError, match="Unknown field kind"):
        p2.merge_material_parts("temperature", v, v)
    with pytest.raises(ValueError, match="equal length"):
        p2.merge_material_parts("density", v, v[:1])


# --------------------------------------------------------------------------
# LSC dataset
# --------------------------------------------------------------------------
def test_lsc_sample_structure_and_sim_time(tmp_path: Path) -> None:
    """LSC samples use canonical channels and sim_time lead times."""
    _write_lsc_triplet(tmp_path, (0.26, 0.49, 0.77))
    ds = p2.LSC_2frame_canonical_DataSet(
        str(tmp_path),
        _prefix_file(tmp_path, "lsc240420_id00001"),
        max_timeIDX_offset=1,
        max_file_checks=1,
        thermo_kinds=("density", "pressure"),
    )
    ds._rng = _FixedRNG([1, 1, 1])  # gaps 1, 1; prev index 1

    x, y, lead, cmap, ds_id = ds[0]

    names = [CANONICAL_FIELDS[i] for i in cmap]
    assert names == [
        "Uvelocity",
        "Wvelocity",
        "density_Cu",
        "pressure_Cu",
        "density_HE",
        "pressure_HE",
        "density_Steel.alloySS304L",
        "pressure_Steel.alloySS304L",
    ]
    assert x.shape == (2, 8, H, W) and y.shape == (8, H, W)
    assert x.dtype == torch.float32 and cmap.dtype == torch.long
    assert torch.allclose(lead, torch.tensor([0.23, 0.28]))
    assert int(ds_id) == DATASET_IDS["lsc"]
    # Throw (row 1) density in frame idx 2 is 2 * 2; NaNs elsewhere became 0.
    cu = names.index("density_Cu")
    assert torch.allclose(x[1, cu, 1], torch.full((W,), 4.0))
    assert torch.all(x[1, cu, 0] == 0)
    assert torch.allclose(y[0], torch.full((H, W), 3.0))


def test_full_image_reflection(tmp_path: Path) -> None:
    """half_image=False reflects each channel to double the width."""
    _write_lsc_triplet(tmp_path, (0.25, 0.5, 0.75))
    ds = p2.LSC_2frame_canonical_DataSet(
        str(tmp_path),
        _prefix_file(tmp_path, "lsc240420_id00001"),
        max_timeIDX_offset=1,
        max_file_checks=1,
        half_image=False,
    )
    ds._rng = _FixedRNG([1, 1, 1])
    x, y, *_ = ds[0]
    assert y.shape[-1] == 2 * W
    assert torch.equal(y[..., :W], torch.flip(y[..., W:], dims=[-1]))


@pytest.mark.parametrize("times", [(0.25, None, 0.75), (0.5, 0.5, 0.75)])
def test_bad_times_are_redrawn(tmp_path: Path, times: tuple) -> None:
    """Missing sim_time or non-increasing times reject the triplet."""
    _write_lsc_triplet(tmp_path, times)
    ds = p2.LSC_2frame_canonical_DataSet(
        str(tmp_path),
        _prefix_file(tmp_path, "lsc240420_id00001"),
        max_timeIDX_offset=1,
        max_file_checks=2,
    )
    ds._rng = _FixedRNG([1, 1, 1] * 10)
    with pytest.raises(RuntimeError, match="no usable sample"):
        ds[0]


def test_missing_frame_moves_to_next_prefix(tmp_path: Path) -> None:
    """A prefix with a missing frame is skipped for the next prefix."""
    _write_lsc_triplet(tmp_path, (0.25, 0.5, 0.75))
    ds = p2.LSC_2frame_canonical_DataSet(
        str(tmp_path),
        _prefix_file(tmp_path, "lsc240420_id00001"),
        max_timeIDX_offset=1,
        max_file_checks=1,
    )
    ds.file_prefix_list = ["lsc240420_id09999", "lsc240420_id00001"]
    ds.Nsamples = 2
    ds._rng = _FixedRNG([1, 1, 1, 1, 1, 1])
    x, *_ = ds[0]
    assert x.shape[0] == 2


def test_constructor_validation(tmp_path: Path) -> None:
    """Invalid offsets and empty prefix lists raise."""
    empty = tmp_path / "empty.txt"
    empty.write_text("\n")
    with pytest.raises(ValueError, match="max_timeIDX_offset"):
        p2.LSC_2frame_canonical_DataSet(str(tmp_path), str(empty), 0, 1)
    with pytest.raises(ValueError, match="No prefixes"):
        p2.LSC_2frame_canonical_DataSet(str(tmp_path), str(empty), 1, 1)


def test_rng_is_lazy(tmp_path: Path) -> None:
    """The RNG is created on first use, not at construction."""
    ds = p2.LSC_2frame_canonical_DataSet(
        str(tmp_path), _prefix_file(tmp_path, "a"), 1, 1
    )
    assert ds._rng is None
    assert isinstance(ds.rng, np.random.Generator)
    assert len(ds) == int(1e6)


# --------------------------------------------------------------------------
# CYL dataset
# --------------------------------------------------------------------------
def _write_design(tmp_path: Path, rows: list[tuple[str, str, str]]) -> str:
    """Write a design CSV with the real file's leading-space formatting."""
    lines = ["key, wallT, wallMat, backMat"]
    lines += [f"{k}, 1.0, {w}, {b}" for k, w, b in rows]
    path = tmp_path / "design.csv"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _cyl_fields(back: str, wall_back_overlap: bool) -> dict[str, np.ndarray]:
    """Return CYL-style fields: wall/maincharge/booster by part, back by name."""
    fields = {"Uvelocity": _const(1.0), "Wvelocity": _const(2.0)}
    vofms = {
        "maincharge": np.array([1.0, 0, 0, 0])[:, None] * np.ones((H, W)),
        "booster": np.array([0, 1.0, 0, 0])[:, None] * np.ones((H, W)),
        "wall": np.array([0, 0, 1.0, 0.5])[:, None] * np.ones((H, W)),
        back: np.array([0, 0, 0, 0.5 if wall_back_overlap else 0])[:, None]
        * np.ones((H, W)),
    }
    values = {"maincharge": 2.0, "booster": 1.5, "wall": 4.0, back: 8.0}
    for part, vofm in vofms.items():
        fields[f"vofm_{part}"] = vofm
        inside = vofm > 0
        for kind, scale in (("density", 1.0), ("pressure", 0.1), ("energy", 0.01)):
            fields[f"{kind}_{part}"] = np.where(inside, values[part] * scale, np.nan)
    return fields


def _write_cyl(tmp_path: Path, key: str, back: str, overlap: bool) -> None:
    """Write CYL frames 0, 2, 3 (index 1 missing, as in real data)."""
    for idx, t in ((0, 0.0), (2, 0.5), (3, 0.75)):
        _frame(
            tmp_path / f"{key}_pvi_idx{idx:05d}.npz",
            t,
            **_cyl_fields(back, overlap),
        )


def _cyl_dataset(tmp_path: Path, key: str, design: str) -> p2.PAGOSA_2frame_DataSet:
    """Build a CYL dataset with all thermo kinds over one prefix."""
    return p2.CYL_2frame_canonical_DataSet(
        str(tmp_path),
        design,
        _prefix_file(tmp_path, key),
        max_timeIDX_offset=2,
        max_file_checks=3,
        thermo_kinds=("density", "energy", "pressure"),
    )


def test_cyl_resolves_materials_from_design(tmp_path: Path) -> None:
    """Wall material comes from wallMat; booster merges into HE; idx 1 skipped."""
    key = "cx241203_id00001"
    _write_cyl(tmp_path, key, back="Ta", overlap=False)
    ds = _cyl_dataset(tmp_path, key, _write_design(tmp_path, [(key, "Cu", "Ta")]))
    # First draw needs idx 1 (missing) -> redraw; second is (0, 2, 3).
    ds._rng = _FixedRNG([1, 1, 0, 2, 1, 0])

    x, y, lead, cmap, ds_id = ds[0]

    names = [CANONICAL_FIELDS[i] for i in cmap]
    assert names == [
        "Uvelocity",
        "Wvelocity",
        "density_Cu",
        "energy_Cu",
        "pressure_Cu",
        "density_HE",
        "energy_HE",
        "pressure_HE",
        "density_Ta",
        "energy_Ta",
        "pressure_Ta",
    ]
    assert torch.allclose(lead, torch.tensor([0.5, 0.25]))
    assert int(ds_id) == DATASET_IDS["cyl"]
    he = names.index("density_HE")
    # Row 0 is maincharge (2.0), row 1 is booster (1.5): one HE channel.
    assert torch.allclose(y[he, 0], torch.full((W,), 2.0))
    assert torch.allclose(y[he, 1], torch.full((W,), 1.5))


def test_cyl_same_wall_and_back_material_merges(tmp_path: Path) -> None:
    """Equal wallMat and backMat merge two parts with the mixed-cell rules."""
    key = "cx241203_id00002"
    _write_cyl(tmp_path, key, back="Al", overlap=True)
    ds = _cyl_dataset(tmp_path, key, _write_design(tmp_path, [(key, "Al", "Al")]))
    ds._rng = _FixedRNG([2, 1, 0])

    _, y, _, cmap, _ = ds[0]

    names = [CANONICAL_FIELDS[i] for i in cmap]
    assert "density_Al" in names and len(names) == 2 + 3 * 2
    row = 3  # wall vf 0.5 (rho 4), back vf 0.5 (rho 8)
    dens = y[names.index("density_Al"), row, 0]
    pres = y[names.index("pressure_Al"), row, 0]
    ener = y[names.index("energy_Al"), row, 0]
    assert torch.isclose(dens, torch.tensor(6.0))
    assert torch.isclose(pres, torch.tensor(0.6))
    assert torch.isclose(ener, torch.tensor((0.04 * 2 + 0.08 * 4) / 6))


def test_cyl_void_parts_are_skipped(tmp_path: Path) -> None:
    """Void wall/back produce only HE and kinematic channels."""
    key = "cx241203_id00003"
    _write_cyl(tmp_path, key, back="Void", overlap=False)
    ds = _cyl_dataset(tmp_path, key, _write_design(tmp_path, [(key, "Void", "Void")]))
    ds._rng = _FixedRNG([2, 1, 0])
    *_, cmap, _ = ds[0]
    names = [CANONICAL_FIELDS[i] for i in cmap]
    assert names == [
        "Uvelocity",
        "Wvelocity",
        "density_HE",
        "energy_HE",
        "pressure_HE",
    ]


def test_cyl_unknown_prefix_raises(tmp_path: Path) -> None:
    """A prefix missing from the design CSV raises a clear KeyError."""
    ds = _cyl_dataset(
        tmp_path, "cx241203_id00009", _write_design(tmp_path, [("other", "Al", "Al")])
    )
    with pytest.raises(KeyError, match="not found in design CSV"):
        ds.part_map("cx241203_id00009")
