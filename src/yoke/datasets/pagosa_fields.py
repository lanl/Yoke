"""Canonical, material-based field vocabulary for PAGOSA hydro datasets.

Several Yoke datasets (``lsc240420`` layered shaped charges, ``cx241203``
cylinder expansion, and hopefully more in the future) are produced by the same
multi-material shock-physics code, PAGOSA, but name their per-material NPZ
fields differently:

- **LSC** names every field by *part*: ``density_case``, ``pressure_throw``,
  ...
- **CYL** mixes conventions. The ``wall``, ``maincharge`` and ``booster`` parts
  keep part names (``density_wall``); the wall's material is recorded in the
  design CSV (``wallMat``). The back material is named by material directly
  (``density_Ta``).

To train a single model on several of these datasets, every field is mapped to
a *canonical* name ``<kind>_<material>`` taken from :data:`CANONICAL_FIELDS`.
That list is the model's ``default_vars``, so a channel's position in it is its
identity inside a trained checkpoint. **The vocabulary is append-only**: new
materials or kinematic fields may only be added at the end.

A dataset describes its own naming with a *part map*, ``{npz_part: material}``,
and :func:`resolve_channel_sources` turns that into the ordered list of
canonical channels a sample provides, together with the NPZ part(s) that feed
each channel. Several parts mapping to one material (e.g. CYL ``maincharge``
and ``booster``, both ``HE``) are merged into a single channel by the dataset.
"""

from collections.abc import Container, Iterable, Mapping
from dataclasses import dataclass

#: Thermodynamic field kinds available per material, in vocabulary order.
THERMO_KINDS: tuple[str, ...] = ("density", "energy", "pressure")

#: Kinematic (material-independent) fields, in vocabulary order.
KINEMATIC_FIELDS: tuple[str, ...] = ("Uvelocity", "Wvelocity")

#: Canonical material names. ``Void`` is deliberately absent: it carries no
#: mass or thermodynamic state, so parts resolving to it are skipped.
CANONICAL_MATERIALS: tuple[str, ...] = (
    "Air",
    "Al",
    "Be",
    "Cu",
    "U.DU",
    "HE",
    "N",
    "Sn",
    "Steel.alloySS304L",
    "Polymer.Sylgard",
    "Ta",
    "Water",
)

#: Material name meaning "no material"; parts mapped to it are ignored.
VOID_MATERIAL = "Void"

#: The full canonical channel vocabulary (APPEND-ONLY; see module docstring).
CANONICAL_FIELDS: tuple[str, ...] = KINEMATIC_FIELDS + tuple(
    f"{kind}_{material}" for material in CANONICAL_MATERIALS for kind in THERMO_KINDS
)

_CANONICAL_INDEX: dict[str, int] = {name: i for i, name in enumerate(CANONICAL_FIELDS)}

#: LSC part name -> canonical material.
LSC_PART_TO_MATERIAL: dict[str, str] = {
    "case": "Steel.alloySS304L",
    "cushion": "Polymer.Sylgard",
    "outside_air": "Air",
    "striker": "Al",
    "throw": "Cu",
    "maincharge": "HE",
}

#: CYL parts whose material is the same in every simulation. The ``wall`` part
#: and the back material vary per simulation and come from the design CSV.
CYL_FIXED_PART_TO_MATERIAL: dict[str, str] = {
    "maincharge": "HE",
    "booster": "HE",
}

#: Stable integer tag per dataset, recorded with every per-sample loss. These
#: values are persisted in record CSVs, so existing entries must not change.
DATASET_IDS: dict[str, int] = {"lsc": 0, "cyl": 1}

#: Inverse of :data:`DATASET_IDS`.
DATASET_NAMES: dict[int, str] = {v: k for k, v in DATASET_IDS.items()}


@dataclass(frozen=True)
class ChannelSource:
    """One canonical output channel and the NPZ parts that feed it.

    Attributes:
        canonical (str): Canonical field name, e.g. ``"density_HE"``.
        kind (str): ``"density"``, ``"energy"``, ``"pressure"``, or
            ``"kinematic"``.
        parts (tuple[str, ...]): NPZ part suffixes merged into this channel,
            e.g. ``("booster", "maincharge")``. Empty for kinematic fields,
            which are read directly by their canonical name.
    """

    canonical: str
    kind: str
    parts: tuple[str, ...]


def canonical_index(name: str) -> int:
    """Return the vocabulary index of a canonical field name.

    Args:
        name (str): Canonical field name.

    Returns:
        int: Index of ``name`` in :data:`CANONICAL_FIELDS`.

    Raises:
        KeyError: If ``name`` is not in the vocabulary.
    """
    try:
        return _CANONICAL_INDEX[name]
    except KeyError:
        raise KeyError(
            f"{name!r} is not a canonical PAGOSA field. Append it to "
            "CANONICAL_FIELDS (never reorder) to add support."
        ) from None


def validate_thermo_kinds(thermo_kinds: Iterable[str]) -> tuple[str, ...]:
    """Check requested thermodynamic kinds and return them in vocabulary order.

    Args:
        thermo_kinds (Iterable[str]): Subset of :data:`THERMO_KINDS`.

    Returns:
        tuple[str, ...]: The requested kinds, de-duplicated and ordered.

    Raises:
        ValueError: If a kind is unknown.
    """
    requested = set(thermo_kinds)
    unknown = requested - set(THERMO_KINDS)
    if unknown:
        raise ValueError(
            f"Unknown thermodynamic kinds {sorted(unknown)}; "
            f"choose from {list(THERMO_KINDS)}."
        )
    return tuple(k for k in THERMO_KINDS if k in requested)


def _required_keys(kind: str, part: str) -> tuple[str, ...]:
    """Return the NPZ keys needed to build ``kind`` from ``part``.

    Every part needs its volume fraction (``vofm_<part>``): densities are
    volume-fraction weighted, and merges weight by it. Energy merges are
    mass-weighted, so energy also needs the part's density.

    Args:
        kind (str): Thermodynamic kind.
        part (str): NPZ part suffix.

    Returns:
        tuple[str, ...]: Required NPZ keys.
    """
    keys = (f"{kind}_{part}", f"vofm_{part}")
    if kind == "energy":
        keys += (f"density_{part}",)
    return keys


def resolve_channel_sources(
    part_map: Mapping[str, str],
    available_keys: Container[str],
    thermo_kinds: Iterable[str] = ("density",),
    kinematic: Iterable[str] = KINEMATIC_FIELDS,
) -> list[ChannelSource]:
    """Resolve a dataset's NPZ fields into ordered canonical channels.

    Args:
        part_map (Mapping[str, str]): ``{npz_part: canonical_material}`` for one
            simulation. Parts mapped to ``Void`` are ignored.
        available_keys (Container[str]): NPZ keys present in *every* frame of
            the sample. Parts lacking any required key are dropped, and a
            channel with no remaining parts is omitted.
        thermo_kinds (Iterable[str]): Thermodynamic kinds to include.
        kinematic (Iterable[str]): Kinematic fields to include (canonical names
            that are also the NPZ keys).

    Returns:
        list[ChannelSource]: Channels in :data:`CANONICAL_FIELDS` order.

    Raises:
        KeyError: If a part maps to a material outside the vocabulary, or a
            kinematic field is not canonical.
        ValueError: If ``thermo_kinds`` contains an unknown kind.
    """
    kinds = validate_thermo_kinds(thermo_kinds)
    sources: list[ChannelSource] = []

    for name in kinematic:
        canonical_index(name)  # Fail loudly on a non-canonical name.
        if name in available_keys:
            sources.append(ChannelSource(name, "kinematic", ()))

    # Group parts by material, sorted so the merge order is deterministic.
    parts_by_material: dict[str, list[str]] = {}
    for part, material in sorted(part_map.items()):
        if material == VOID_MATERIAL:
            continue
        if material not in CANONICAL_MATERIALS:
            raise KeyError(
                f"Part {part!r} maps to unknown material {material!r}. Append it "
                "to CANONICAL_MATERIALS (never reorder) to add support."
            )
        parts_by_material.setdefault(material, []).append(part)

    for material, parts in parts_by_material.items():
        for kind in kinds:
            present = tuple(
                p
                for p in parts
                if all(k in available_keys for k in _required_keys(kind, p))
            )
            if present:
                sources.append(ChannelSource(f"{kind}_{material}", kind, present))

    sources.sort(key=lambda s: canonical_index(s.canonical))
    return sources
