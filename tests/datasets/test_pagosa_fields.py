"""Tests for the canonical PAGOSA field vocabulary."""

import pytest

from yoke.datasets import pagosa_fields as pf


def test_vocabulary_order_is_pinned() -> None:
    """The vocabulary prefix is fixed; reordering would break checkpoints."""
    assert pf.CANONICAL_FIELDS[:8] == (
        "Uvelocity",
        "Wvelocity",
        "density_Air",
        "energy_Air",
        "pressure_Air",
        "density_Al",
        "energy_Al",
        "pressure_Al",
    )
    assert len(pf.CANONICAL_FIELDS) == 2 + 3 * len(pf.CANONICAL_MATERIALS)
    assert pf.canonical_index("density_HE") == 2 + 3 * 5
    assert len(set(pf.CANONICAL_FIELDS)) == len(pf.CANONICAL_FIELDS)


def test_void_and_booster_not_canonical() -> None:
    """Void is excluded and booster is folded into HE."""
    assert "Void" not in pf.CANONICAL_MATERIALS
    assert pf.CYL_FIXED_PART_TO_MATERIAL == {"maincharge": "HE", "booster": "HE"}
    assert pf.LSC_PART_TO_MATERIAL["maincharge"] == "HE"
    with pytest.raises(KeyError, match="not a canonical"):
        pf.canonical_index("density_booster")


def test_lsc_parts_map_to_canonical_materials() -> None:
    """Every LSC part maps to a material in the vocabulary."""
    assert set(pf.LSC_PART_TO_MATERIAL.values()) <= set(pf.CANONICAL_MATERIALS)


def test_dataset_ids_round_trip() -> None:
    """Dataset IDs are stable and invertible."""
    assert pf.DATASET_IDS == {"lsc": 0, "cyl": 1}
    assert pf.DATASET_NAMES[1] == "cyl"


def _keys(parts: list[str], kinds: tuple[str, ...]) -> set[str]:
    """Build the NPZ key set for ``parts`` with ``kinds`` and volume fractions."""
    keys = {"Uvelocity", "Wvelocity"}
    for p in parts:
        keys.add(f"vofm_{p}")
        keys.update(f"{k}_{p}" for k in kinds)
    return keys


def test_resolve_merges_parts_and_orders_channels() -> None:
    """booster+maincharge merge into HE; output follows vocabulary order."""
    part_map = {"wall": "Polymer.Sylgard", "Ta": "Ta", **pf.CYL_FIXED_PART_TO_MATERIAL}
    keys = _keys(["wall", "Ta", "booster", "maincharge"], pf.THERMO_KINDS)

    sources = pf.resolve_channel_sources(part_map, keys, thermo_kinds=pf.THERMO_KINDS)

    names = [s.canonical for s in sources]
    assert names[:2] == ["Uvelocity", "Wvelocity"]
    assert names == sorted(names, key=pf.canonical_index)
    he = {s.canonical: s for s in sources}["energy_HE"]
    assert he.parts == ("booster", "maincharge")
    assert len(sources) == 2 + 3 * 3


def test_resolve_skips_void_and_missing_fields() -> None:
    """Void parts are ignored; parts lacking required keys are dropped."""
    part_map = {"wall": "Void", "Cu": "Cu", "maincharge": "HE", "booster": "HE"}
    keys = _keys(["Cu", "maincharge"], ("density", "pressure"))
    keys.add("density_booster")  # but no vofm_booster -> booster dropped

    sources = pf.resolve_channel_sources(
        part_map, keys, thermo_kinds=("energy", "density"), kinematic=("Uvelocity",)
    )

    by_name = {s.canonical: s for s in sources}
    # Energy is requested but absent (and needs density + vofm too).
    assert set(by_name) == {"Uvelocity", "density_Cu", "density_HE"}
    assert by_name["density_HE"].parts == ("maincharge",)


def test_resolve_requires_density_for_energy() -> None:
    """An energy channel needs the part's density for mass weighting."""
    keys = {"energy_x", "vofm_x"}
    assert pf.resolve_channel_sources({"x": "Cu"}, keys, ("energy",), ()) == []
    keys.add("density_x")
    (src,) = pf.resolve_channel_sources({"x": "Cu"}, keys, ("energy",), ())
    assert src.canonical == "energy_Cu"


def test_resolve_rejects_unknown_names() -> None:
    """Unknown materials, kinds, or kinematic fields raise."""
    with pytest.raises(KeyError, match="unknown material"):
        pf.resolve_channel_sources({"x": "Unobtainium"}, set())
    with pytest.raises(ValueError, match="Unknown thermodynamic"):
        pf.resolve_channel_sources({}, set(), thermo_kinds=("temperature",))
    with pytest.raises(KeyError, match="not a canonical"):
        pf.resolve_channel_sources({}, set(), kinematic=("Rcoord",))
