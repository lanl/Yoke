"""Fast unit tests for yoke.datasets.load_npz_dataset.

These tests avoid expensive filesystem probing and long retry loops by patching
TemporalDataSet._build_valid_prefixes and using tiny NPZ fixtures.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import torch

import yoke.datasets.load_npz_dataset as m


def _write_npz(path: pathlib.Path, **fields: np.ndarray) -> None:
    """Write a small NPZ file with provided fields."""
    np.savez(path, **fields)


def _write_prefix_file(path: pathlib.Path, prefixes: list[str]) -> None:
    """Write a newline-delimited prefix file."""
    path.write_text("\n".join(prefixes) + "\n", encoding="utf-8")


def _write_design_csv(path: pathlib.Path, rows: list[tuple[str, str, str]]) -> None:
    """Write a minimal design CSV with idx, wallMat, backMat columns."""
    lines = ["idx,wallMat,backMat"]
    lines += [f"{idx},{wall},{back}" for idx, wall, back in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class _FakeRNG:
    """Deterministic RNG used to control dataset index selection in tests."""

    def __init__(self, values: list[int]) -> None:
        """Initialize with a fixed queue of integers."""
        self._values = list(values)

    def integers(self, low: int, high: int, *, endpoint: bool = False) -> int:
        """Return the next queued integer."""
        _ = (low, high, endpoint)
        if not self._values:
            raise RuntimeError("FakeRNG queue exhausted")
        return int(self._values.pop(0))


@pytest.fixture(autouse=True)
def _disable_temporal_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable expensive TemporalDataSet prefix probing in unit tests."""
    monkeypatch.setattr(m.TemporalDataSet, "_build_valid_prefixes", lambda self: None)


def test_current_rank_returns_zero_when_dist_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_current_rank returns 0 when torch.distributed is unavailable."""
    monkeypatch.setattr(m, "dist", None)
    assert m._current_rank() == 0


def test_rank_worker_tag_contains_rank_worker_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rank_worker_tag includes rank/worker/pid and optional index."""

    class _Info:
        id = 7

    monkeypatch.setattr(m, "get_worker_info", lambda: _Info())
    monkeypatch.setattr(m, "_current_rank", lambda: 3)
    tag = m.rank_worker_tag(11)
    assert "rank3" in tag
    assert "worker7" in tag
    assert "pid" in tag
    assert "idx=11" in tag


@pytest.mark.parametrize(
    ("s", "expected"),
    [
        ("density_Air", True),
        ("density_", True),
        ("pressure_Air", False),
        ("", False),
    ],
)
def test_has_density_prefix(s: str, expected: bool) -> None:
    """has_density_prefix detects density_* fields."""
    assert m.has_density_prefix(s) is expected


@pytest.mark.parametrize(
    ("s", "expected"),
    [
        ("density_Air", "Air"),
        ("density_", ""),
        ("pressure_Air", None),
        ("", None),
    ],
)
def test_extract_after_density(s: str, expected: str | None) -> None:
    """extract_after_density returns suffix after density_ or None."""
    assert m.extract_after_density(s) == expected


def test_read_npz_nan_from_path_and_replaces_nan(tmp_path: pathlib.Path) -> None:
    """read_npz_nan loads a field and replaces NaN with 0."""
    p = tmp_path / "a.npz"
    arr = np.array([[np.nan, 2.0]], dtype=float)
    _write_npz(p, field=arr)
    out = m.read_npz_nan(p, "field")
    assert out.shape == (1, 2)
    assert out[0, 0] == 0.0
    assert out[0, 1] == 2.0


def test_read_npz_nan_missing_field_raises(tmp_path: pathlib.Path) -> None:
    """read_npz_nan raises KeyError when the field is absent."""
    p = tmp_path / "a.npz"
    _write_npz(p, other=np.zeros((1,), dtype=float))
    with pytest.raises(KeyError):
        _ = m.read_npz_nan(p, "field")


def test_read_npz_nan_invalid_type_raises() -> None:
    """read_npz_nan raises TypeError on unsupported npz input types."""
    with pytest.raises(TypeError):
        _ = m.read_npz_nan(123, "field")  # type: ignore[arg-type]


def test_handle_voids_returns_none_when_not_void(tmp_path: pathlib.Path) -> None:
    """handle_voids returns None when hfield does not end with _Void."""
    p = tmp_path / "a.npz"
    _write_npz(p, av_density=np.zeros((2, 2)))
    assert m.handle_voids(p, "density_Air") is None


def test_handle_voids_sets_nan_where_any_density_present(tmp_path: pathlib.Path) -> None:
    """handle_voids returns an all-NaN mask for current read_npz_nan semantics."""
    p = tmp_path / "a.npz"
    av = np.zeros((2, 2), dtype=float)

    # Note: read_npz_nan converts NaNs to 0, so mask becomes True everywhere.
    booster = np.array([[np.nan, 1.0], [np.nan, np.nan]], dtype=float)
    main = np.array([[np.nan, np.nan], [2.0, np.nan]], dtype=float)
    wall = np.array([[np.nan, np.nan], [np.nan, 3.0]], dtype=float)

    _write_npz(
        p,
        av_density=av,
        density_booster=booster,
        density_maincharge=main,
        density_wall=wall,
    )

    out = m.handle_voids(p, "density_Void")
    assert out is not None
    assert out.shape == (2, 2)
    assert np.all(np.isnan(out))


def test_meshgrid_position_rcoord(tmp_path: pathlib.Path) -> None:
    """meshgrid_position expands Rcoord into a 2D mesh."""
    p = tmp_path / "a.npz"
    r = np.array([1.0, 2.0])
    z = np.array([10.0, 20.0, 30.0])
    _write_npz(p, Rcoord=r, Zcoord=z)
    out = m.meshgrid_position(r, p, "Rcoord")
    assert out.shape == (3, 2)
    assert np.all(out[0, :] == r)


def test_meshgrid_position_zcoord(tmp_path: pathlib.Path) -> None:
    """meshgrid_position expands Zcoord into a 2D mesh."""
    p = tmp_path / "a.npz"
    r = np.array([1.0, 2.0])
    z = np.array([10.0, 20.0, 30.0])
    _write_npz(p, Rcoord=r, Zcoord=z)
    out = m.meshgrid_position(z, p, "Zcoord")
    assert out.shape == (3, 2)
    assert np.all(out[:, 0] == z)


def test_volfrac_density_no_density_prefix_returns_input(tmp_path: pathlib.Path) -> None:
    """volfrac_density does nothing for non-density fields."""
    p = tmp_path / "a.npz"
    _write_npz(p, pressure_Air=np.ones((2, 2)))
    img = np.ones((2, 2))
    out = m.volfrac_density(img, p, "pressure_Air")
    assert np.all(out == img)


def test_volfrac_density_empty_suffix_prints_and_returns_input(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """volfrac_density prints a warning for 'density_' and returns the input."""
    p = tmp_path / "a.npz"
    _write_npz(p, density_=np.ones((2, 2)))
    img = np.ones((2, 2))
    out = m.volfrac_density(img, p, "density_")
    captured = capsys.readouterr()
    assert "Could not extract suffix from hfield" in captured.out
    assert np.all(out == img)


def test_volfrac_density_multiplies_by_vofm(tmp_path: pathlib.Path) -> None:
    """volfrac_density multiplies density by the matching vofm_* field."""
    p = tmp_path / "a.npz"
    dens = np.array([[2.0, 2.0]], dtype=float)
    vofm = np.array([[0.5, 0.25]], dtype=float)
    _write_npz(p, density_Air=dens, vofm_Air=vofm)
    out = m.volfrac_density(dens, p, "density_Air")
    assert np.allclose(out, dens * vofm)


def test_import_img_from_npz_applies_meshgrid_and_volfrac(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """import_img_from_npz applies void handling, meshgrid, and volfrac transforms."""
    monkeypatch.setattr(m, "handle_voids", lambda npz, fld: None)
    monkeypatch.setattr(m, "read_npz_nan", lambda npz, fld: np.ones((2, 2)))
    monkeypatch.setattr(m, "meshgrid_position", lambda img, npz, fld: img + 1)
    monkeypatch.setattr(m, "volfrac_density", lambda img, npz, fld: img * 2)

    out = m.import_img_from_npz("x.npz", "density_Air")
    assert np.all(out == (np.ones((2, 2)) + 1) * 2)


def test_combine_by_number_and_label_combines_duplicates() -> None:
    """combine_by_number_and_label merges repeated channel numbers."""
    nums = [0, 0, 1]
    labels = ["A", "A2", "B"]
    arr = np.zeros((3, 2, 2), dtype=float)
    arr[0, 0, 0] = 1.0
    arr[1, 0, 1] = 2.0
    arr[2, 1, 1] = 3.0

    u_nums, comb, u_labels = m.combine_by_number_and_label(nums, arr, labels)
    assert set(u_nums.tolist()) == {0, 1}

    idx0 = int(np.where(u_nums == 0)[0][0])
    assert comb[idx0, 0, 0] == 1.0
    assert comb[idx0, 0, 1] == 2.0
    assert u_labels[idx0] in ("A", "A2")


def test_combine_by_number_and_label_mismatch_raises() -> None:
    """combine_by_number_and_label raises ValueError on inconsistent lengths."""
    with pytest.raises(ValueError):
        _ = m.combine_by_number_and_label([0], np.zeros((2, 2, 2)), ["A"])


def test_process_channel_data_no_duplicates_returns_input() -> None:
    """process_channel_data returns inputs unchanged when channels are unique."""
    cm = [0, 1]
    imgs = np.zeros((2, 2, 2, 2))
    names = ["A", "B"]
    cm2, imgs2, names2 = m.process_channel_data(cm, imgs, names)
    assert cm2 == cm
    assert imgs2 is imgs
    assert names2 == names


def test_process_channel_data_with_duplicates_combines() -> None:
    """process_channel_data combines duplicate channels and reduces channel count."""
    cm = [0, 0]
    names = ["A", "A2"]
    imgs = np.zeros((1, 2, 2, 2), dtype=float)
    imgs[0, 0, 0, 0] = 1.0
    imgs[0, 1, 0, 1] = 2.0

    cm2, imgs2, names2 = m.process_channel_data(cm, imgs, names)
    assert cm2 == [0]
    assert imgs2.shape == (1, 1, 2, 2)
    assert names2[0] in ("A", "A2")
    assert imgs2[0, 0, 0, 0] == 1.0
    assert imgs2[0, 0, 0, 1] == 2.0


def test_labeled_data_study_key_extraction(tmp_path: pathlib.Path) -> None:
    """LabeledData extracts key/study and produces an active channel map."""
    csv = tmp_path / "design.csv"
    _write_design_csv(csv, [("cx241203_id00001", "Air", "Al")])

    npz = tmp_path / "cx241203_id00001_pvi_idx00000.npz"
    _write_npz(npz, dummy=np.zeros((1,), dtype=float))

    ld = m.LabeledData(npz, csv)
    assert ld.key == "cx241203_id00001"
    assert ld.study == "cx"
    assert isinstance(ld.get_channel_map(), list)


@pytest.mark.parametrize("kv", ["velocity", "position", "both"])
def test_labeled_data_kinematic_modes(tmp_path: pathlib.Path, kv: str) -> None:
    """LabeledData configures active kinematic fields by mode."""
    csv = tmp_path / "design.csv"
    _write_design_csv(csv, [("cx241203_id00001", "Air", "Al")])

    npz = tmp_path / "cx241203_id00001_pvi_idx00000.npz"
    _write_npz(npz, dummy=np.zeros((1,), dtype=float))

    ld = m.LabeledData(npz, csv, kinematic_variables=kv)
    fields = ld.get_active_npz_field_names()
    if kv == "velocity":
        assert fields[:2] == ["Uvelocity", "Wvelocity"]
    elif kv == "position":
        assert fields[:2] == ["Rcoord", "Zcoord"]
    else:
        assert fields[:4] == ["Rcoord", "Zcoord", "Uvelocity", "Wvelocity"]


def test_labeled_data_invalid_kinematic_raises(tmp_path: pathlib.Path) -> None:
    """LabeledData raises on invalid kinematic_variables."""
    csv = tmp_path / "design.csv"
    _write_design_csv(csv, [("cx241203_id00001", "Air", "Al")])

    npz = tmp_path / "cx241203_id00001_pvi_idx00000.npz"
    _write_npz(npz, dummy=np.zeros((1,), dtype=float))

    with pytest.raises(ValueError, match="kinematic_variables"):
        _ = m.LabeledData(npz, csv, kinematic_variables="nope")


def test_temporal_dataset_len_is_constant(tmp_path: pathlib.Path) -> None:
    """TemporalDataSet has a fixed training length."""
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    prefix_file = tmp_path / "prefixes.txt"
    _write_prefix_file(prefix_file, ["cx241203_id00001"])
    csv = tmp_path / "design.csv"
    _write_design_csv(csv, [("cx241203_id00001", "Air", "Al")])

    ds = m.TemporalDataSet(
        npz_dir=str(npz_dir) + "/",
        csv_filepath=str(csv),
        file_prefix_list=str(prefix_file),
        max_timeIDX_offset=1,
        max_file_checks=1,
        half_image=True,
        kinematic_variables="velocity",
        thermodynamic_variables="density",
    )
    assert len(ds) == 800_000


def test_temporal_dataset_getitem_success_minimal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """TemporalDataSet returns the expected tuple types/shapes in a minimal success."""
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    prefix_file = tmp_path / "prefixes.txt"
    _write_prefix_file(prefix_file, ["cx241203_id00001"])
    csv = tmp_path / "design.csv"
    _write_design_csv(csv, [("cx241203_id00001", "Air", "Al")])

    start = npz_dir / "cx241203_id00001_pvi_idx00000.npz"
    end = npz_dir / "cx241203_id00001_pvi_idx00001.npz"
    _write_npz(start, dummy=np.zeros((1,), dtype=float))
    _write_npz(end, dummy=np.zeros((1,), dtype=float))

    class FakeLabeledData:
        """Minimal stub that matches the tiny NPZ fields used in this test."""

        def __init__(
            self,
            npz_filepath: str,
            csv_filepath: str,
            kinematic_variables: str = "velocity",
            thermodynamic_variables: str = "density",
        ) -> None:
            _ = (
                npz_filepath,
                csv_filepath,
                kinematic_variables,
                thermodynamic_variables,
            )

        def get_active_npz_field_names(self) -> list[str]:
            """Return present fields."""
            return ["dummy"]

        def get_active_hydro_field_names(self) -> list[str]:
            """Return names for present fields."""
            return ["dummy"]

        def get_channel_map(self) -> list[int]:
            """Return a single channel index."""
            return [0]

        def get_all_hydro_field_names(self) -> list[str]:
            """Return the full list of hydro field names."""
            return ["dummy"]

    monkeypatch.setattr(m, "LabeledData", FakeLabeledData)
    monkeypatch.setattr(m, "import_img_from_npz", lambda npz, fld: np.ones((2, 2)))
    monkeypatch.setattr(
        m,
        "process_channel_data",
        lambda cm, imgs, names: (cm, imgs, names),
    )

    ds = m.TemporalDataSet(
        npz_dir=str(npz_dir) + "/",
        csv_filepath=str(csv),
        file_prefix_list=str(prefix_file),
        max_timeIDX_offset=1,
        max_file_checks=1,
        half_image=True,
        kinematic_variables="velocity",
        thermodynamic_variables="density",
    )
    ds.rng = _FakeRNG([1, 0])  # seq_len=1, start_idx=0 -> end_idx=1

    start_img, cm1, end_img, cm2, dt = ds[0]
    assert start_img.shape == (1, 2, 2)
    assert end_img.shape == (1, 2, 2)
    assert cm1.tolist() == [0]
    assert cm2.tolist() == [0]
    assert dt.item() == pytest.approx(0.25)


class _InfiniteRNG:
    """RNG stub that never exhausts (always returns a constant)."""

    def __init__(self, value: int = 0) -> None:
        """Initialize with a constant return value."""
        self._value = value

    def integers(self, low: int, high: int, *, endpoint: bool = False) -> int:
        """Return a constant integer within the requested range."""
        _ = (low, high, endpoint)
        return self._value


def test_sequential_dataset_init_missing_dir_raises(tmp_path: pathlib.Path) -> None:
    """SequentialDataSet raises FileNotFoundError if npz_dir does not exist."""
    prefix_file = tmp_path / "prefixes.txt"
    _write_prefix_file(prefix_file, ["cx241203_id00001"])
    csv = tmp_path / "design.csv"
    _write_design_csv(csv, [("cx241203_id00001", "Air", "Al")])

    with pytest.raises(FileNotFoundError):
        _ = m.SequentialDataSet(
            npz_dir=str(tmp_path / "nope"),
            csv_filepath=str(csv),
            file_prefix_list=str(prefix_file),
            max_file_checks=1,
            seq_len=2,
            kinematic_variables="velocity",
            thermodynamic_variables="density",
        )


def test_sequential_dataset_getitem_success_minimal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """SequentialDataSet returns (seq, dt, channel_map) for a minimal success case."""
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    prefix_file = tmp_path / "prefixes.txt"
    _write_prefix_file(prefix_file, ["cx241203_id00001"])
    csv = tmp_path / "design.csv"
    _write_design_csv(csv, [("cx241203_id00001", "Air", "Al")])

    f0 = npz_dir / "cx241203_id00001_pvi_idx00000.npz"
    f1 = npz_dir / "cx241203_id00001_pvi_idx00001.npz"
    _write_npz(f0, dummy=np.zeros((1,), dtype=float))
    _write_npz(f1, dummy=np.zeros((1,), dtype=float))

    class FakeLabeledData:
        """Minimal stub that matches the tiny NPZ fields used in this test."""

        def __init__(
            self,
            npz_filepath: str,
            csv_filepath: str,
            kinematic_variables: str = "velocity",
            thermodynamic_variables: str = "density",
        ) -> None:
            _ = (
                npz_filepath,
                csv_filepath,
                kinematic_variables,
                thermodynamic_variables,
            )

        def get_active_npz_field_names(self) -> list[str]:
            """Return present fields."""
            return ["dummy"]

        def get_active_hydro_field_names(self) -> list[str]:
            """Return names for present fields."""
            return ["dummy"]

        def get_channel_map(self) -> list[int]:
            """Return a single channel index."""
            return [0]

    monkeypatch.setattr(m, "LabeledData", FakeLabeledData)
    monkeypatch.setattr(m, "import_img_from_npz", lambda npz, fld: np.ones((2, 2)))
    monkeypatch.setattr(
        m,
        "process_channel_data",
        lambda cm, imgs, names: (cm, imgs, names),
    )

    ds = m.SequentialDataSet(
        npz_dir=str(npz_dir),
        csv_filepath=str(csv),
        file_prefix_list=str(prefix_file),
        max_file_checks=1,
        seq_len=2,
        half_image=True,
        kinematic_variables="velocity",
        thermodynamic_variables="density",
    )
    ds.rng = _FakeRNG([0])  # start_idx=0

    img_seq, dt, cm = ds[0]
    assert isinstance(img_seq, torch.Tensor)
    assert img_seq.shape == (2, 1, 2, 2)
    assert dt.item() == pytest.approx(0.25)
    assert cm == [0]


def test_handle_voids_returns_none_for_non_void_field(tmp_path: pathlib.Path) -> None:
    """handle_voids returns None when hfield does not end with '_Void'."""
    p = tmp_path / "a.npz"
    np.savez(
        p,
        av_density=np.zeros((2, 2), dtype=float),
        density_booster=np.zeros((2, 2), dtype=float),
        density_maincharge=np.zeros((2, 2), dtype=float),
    )

    assert m.handle_voids(p, "density_Air") is None


def test_handle_voids_returns_nan_mask_for_void_field(tmp_path: pathlib.Path) -> None:
    """handle_voids returns a NaN mask for '_Void' fields.

    read_npz_nan replaces NaNs with 0, so the internal mask becomes True
    everywhere when the booster/maincharge fields exist.
    """
    p = tmp_path / "a.npz"
    booster = np.array([[np.nan, 1.0], [0.0, np.nan]], dtype=float)
    main = np.array([[np.nan, 0.0], [2.0, np.nan]], dtype=float)
    np.savez(
        p,
        av_density=np.zeros((2, 2), dtype=float),
        density_booster=booster,
        density_maincharge=main,
    )

    out = m.handle_voids(p, "density_Void")
    assert out is not None
    assert out.shape == (2, 2)
    assert np.all(np.isnan(out))


def test_read_npz_nan_accepts_npz_handle(tmp_path: pathlib.Path) -> None:
    """read_npz_nan accepts an opened NpzFile handle."""
    p = tmp_path / "a.npz"
    np.savez(p, field=np.array([[np.nan, 2.0]], dtype=float))

    with np.load(p, allow_pickle=False) as z:
        out = m.read_npz_nan(z, "field")

    assert out[0, 0] == 0.0
    assert out[0, 1] == 2.0


def test_meshgrid_position_noop_for_other_fields(tmp_path: pathlib.Path) -> None:
    """meshgrid_position returns input unchanged for non-coordinate fields."""
    p = tmp_path / "a.npz"
    np.savez(p, dummy=np.zeros((1,), dtype=float))

    img = np.ones((2, 2), dtype=float)
    out = m.meshgrid_position(img, p, "density_Air")
    assert out is img


def test_import_img_from_npz_uses_handle_voids_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """import_img_from_npz uses handle_voids output when non-None."""
    monkeypatch.setattr(m, "handle_voids", lambda npz, fld: np.full((2, 2), 3.0))
    monkeypatch.setattr(m, "read_npz_nan", lambda npz, fld: np.full((2, 2), 9.0))
    monkeypatch.setattr(m, "meshgrid_position", lambda img, npz, fld: img)
    monkeypatch.setattr(m, "volfrac_density", lambda img, npz, fld: img)

    out = m.import_img_from_npz("x.npz", "density_Void")
    assert np.all(out == 3.0)


def test_temporal_dataset_init_with_nondefault_variable_modes(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TemporalDataSet stores non-default variable mode settings."""
    monkeypatch.setattr(
        m.TemporalDataSet,
        "_build_valid_prefixes",
        lambda self: None,
    )

    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    prefix_file = tmp_path / "prefixes.txt"
    prefix_file.write_text("cx241203_id00001\n", encoding="utf-8")
    csv_path = tmp_path / "design.csv"
    csv_path.write_text(
        "idx,wallMat,backMat\ncx241203_id00001,Air,Al\n",
        encoding="utf-8",
    )

    ds = m.TemporalDataSet(
        npz_dir=str(npz_dir) + "/",
        csv_filepath=str(csv_path),
        file_prefix_list=str(prefix_file),
        max_timeIDX_offset=1,
        max_file_checks=1,
        kinematic_variables="both",
        thermodynamic_variables="density and pressure",
        half_image=True,
    )

    assert ds.kinematic_variables == "both"
    assert ds.thermodynamic_variables == "density and pressure"


def test_sequential_dataset_init_with_nondefault_variable_modes(
    tmp_path: pathlib.Path,
) -> None:
    """SequentialDataSet stores non-default variable mode settings."""
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    prefix_file = tmp_path / "prefixes.txt"
    prefix_file.write_text("cx241203_id00001\n", encoding="utf-8")
    csv_path = tmp_path / "design.csv"
    csv_path.write_text(
        "idx,wallMat,backMat\ncx241203_id00001,Air,Al\n",
        encoding="utf-8",
    )

    ds = m.SequentialDataSet(
        npz_dir=str(npz_dir),
        csv_filepath=str(csv_path),
        file_prefix_list=str(prefix_file),
        max_file_checks=1,
        seq_len=2,
        kinematic_variables="position",
        thermodynamic_variables="density and energy",
        half_image=True,
    )

    assert ds.kinematic_variables == "position"
    assert ds.thermodynamic_variables == "density and energy"
