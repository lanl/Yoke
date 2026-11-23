"""Two-frame, canonical-channel datasets for PAGOSA hydro simulations.

These datasets feed the 2-in/1-out :class:`~yoke.models.vit.swin.bomberman.
LodeRunnerViT` from any PAGOSA dataset, using the shared material-based
vocabulary in :mod:`yoke.datasets.pagosa_fields`. A model trained with
``default_vars=CANONICAL_FIELDS`` can then consume LSC and CYL samples
interchangeably (see :class:`yoke.datasets.mixed_dataset.MixedDataSet`).

Every sample is a 5-tuple::

    (input_frames, target_frame, lead_times, channel_map, dataset_id)

- ``input_frames``: ``(2, C, H, W)`` float32, ordered ``[x_{t-1}, x_t]``.
- ``target_frame``: ``(C, H, W)`` float32, ``x_{t+1}``.
- ``lead_times``: ``(2,)`` float32 ``(dt_in, dt_out)``, taken from each
  frame's ``sim_time`` (not the approximate ``0.25 * index`` spacing).
- ``channel_map``: ``(C,)`` int64 indices into ``CANONICAL_FIELDS``. Its
  length varies between simulations, so batches must hold samples sharing one
  map (batch size 1 is always safe).
- ``dataset_id``: int64 scalar from ``DATASET_IDS``, used to tag loss records.

Field construction follows the existing Yoke loaders: NaNs (outside a
material) become ``0`` and densities are volume-fraction weighted
(``density_X * vofm_X``). When several NPZ parts map to one material they are
merged per cell by :func:`merge_material_parts`.
"""

import random
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from yoke.datasets.pagosa_fields import (
    CYL_FIXED_PART_TO_MATERIAL,
    DATASET_IDS,
    KINEMATIC_FIELDS,
    LSC_PART_TO_MATERIAL,
    ChannelSource,
    canonical_index,
    resolve_channel_sources,
    validate_thermo_kinds,
)

#: Highest file index in a simulation's time series (files ``idx00000``-``99``).
MAX_FILE_IDX = 99


def merge_material_parts(
    kind: str,
    values: list[np.ndarray],
    vofms: list[np.ndarray],
    densities: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Merge one field kind from several parts of the same material.

    All inputs are NaN-free (NaN already replaced by ``0``) and ``densities``
    are the *raw* material densities, not volume-fraction weighted.

    - ``density``: sum of volume-fraction-weighted densities,
      ``sum_i rho_i * vf_i``. This is the exact mass per unit volume.
    - ``pressure``: volume-fraction-weighted mean,
      ``sum_i p_i * vf_i / sum_i vf_i``.
    - ``energy`` (specific): mass-weighted mean,
      ``sum_i e_i * rho_i * vf_i / sum_i rho_i * vf_i``.

    Cells with zero total weight are ``0``. The pressure and energy rules are a
    provisional choice pending confirmation of what PAGOSA itself does in mixed
    cells; they only matter where parts of one material overlap.

    A single part is passed through unchanged (``rho * vf`` for density, raw
    values otherwise), which matches the existing single-material loaders.

    Args:
        kind (str): ``"density"``, ``"energy"`` or ``"pressure"``.
        values (list[np.ndarray]): Per-part field values.
        vofms (list[np.ndarray]): Per-part volume fractions.
        densities (list[np.ndarray] | None): Per-part raw densities; required
            for ``"energy"`` when more than one part is merged.

    Returns:
        np.ndarray: The merged field.

    Raises:
        ValueError: If ``kind`` is unknown or required inputs are missing.
    """
    if len(values) != len(vofms) or not values:
        raise ValueError("values and vofms must be non-empty and equal length.")

    if kind == "density":
        return sum(v * f for v, f in zip(values, vofms))
    if len(values) == 1:
        return values[0]

    if kind == "pressure":
        weights = vofms
    elif kind == "energy":
        if densities is None or len(densities) != len(values):
            raise ValueError("Merging energy requires a density per part.")
        weights = [r * f for r, f in zip(densities, vofms)]
    else:
        raise ValueError(f"Unknown field kind {kind!r}.")

    numer = sum(v * w for v, w in zip(values, weights))
    denom = sum(weights)
    out = np.zeros_like(numer)
    np.divide(numer, denom, out=out, where=denom > 0)
    return out


def _nan0(npz: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    """Read an NPZ array with NaNs replaced by ``0``.

    Args:
        npz (np.lib.npyio.NpzFile): Open NPZ file.
        key (str): Field name.

    Returns:
        np.ndarray: The field with NaNs set to ``0``.
    """
    return np.nan_to_num(npz[key], nan=0.0)


def build_channel(npz: np.lib.npyio.NpzFile, source: ChannelSource) -> np.ndarray:
    """Build one canonical channel image from an open NPZ frame.

    Args:
        npz (np.lib.npyio.NpzFile): Open NPZ frame.
        source (ChannelSource): Channel description from
            :func:`~yoke.datasets.pagosa_fields.resolve_channel_sources`.

    Returns:
        np.ndarray: The 2-D channel image.
    """
    if source.kind == "kinematic":
        return _nan0(npz, source.canonical)

    values = [_nan0(npz, f"{source.kind}_{p}") for p in source.parts]
    vofms = [_nan0(npz, f"vofm_{p}") for p in source.parts]
    densities = None
    if source.kind == "energy":
        densities = [_nan0(npz, f"density_{p}") for p in source.parts]
    return merge_material_parts(source.kind, values, vofms, densities)


class PAGOSA_2frame_DataSet(Dataset):
    """Base 2-in/1-out dataset producing canonical PAGOSA channels.

    Subclasses define where a simulation's files live (:meth:`frame_path`) and
    which NPZ parts hold which materials (:meth:`part_map`). Sampling mirrors
    :class:`~yoke.datasets.lsc_dataset.LSC_rho2rho_temporal_2frame_DataSet`:
    two index gaps are drawn independently from ``[1, max_timeIDX_offset]``
    and the triplet is anchored so all indices land in ``[0, 99]``. Triplets
    with a missing file, a missing ``sim_time``, or non-increasing times are
    redrawn.

    Args:
        npz_dir (str): Directory containing the NPZ files.
        file_prefix_list (str): Text file listing one simulation prefix per line.
        max_timeIDX_offset (int): Maximum file-index gap between frames.
        max_file_checks (int): Triplet draws per prefix before moving on.
        half_image (bool): If False, images are reflected about the symmetry
            axis to form full images.
        thermo_kinds (Iterable[str]): Thermodynamic kinds to include.
        kinematic (Iterable[str]): Kinematic fields to include.
    """

    #: Dataset tag; subclasses set this to a key of ``DATASET_IDS``.
    dataset_name: str = ""

    def __init__(
        self,
        npz_dir: str,
        file_prefix_list: str,
        max_timeIDX_offset: int,
        max_file_checks: int,
        half_image: bool = True,
        thermo_kinds: Iterable[str] = ("density",),
        kinematic: Iterable[str] = KINEMATIC_FIELDS,
    ) -> None:
        """Initialize the dataset and read the prefix list."""
        if max_timeIDX_offset < 1 or 2 * max_timeIDX_offset > MAX_FILE_IDX:
            raise ValueError(
                f"max_timeIDX_offset must be in [1, {MAX_FILE_IDX // 2}]; "
                f"got {max_timeIDX_offset}."
            )
        self.npz_dir = Path(npz_dir)
        self.max_timeIDX_offset = max_timeIDX_offset
        self.max_file_checks = max_file_checks
        self.half_image = half_image
        self.thermo_kinds = validate_thermo_kinds(thermo_kinds)
        self.kinematic = tuple(kinematic)
        self.dataset_id = DATASET_IDS[self.dataset_name]

        with open(file_prefix_list) as f:
            self.file_prefix_list = [line.strip() for line in f if line.strip()]
        if not self.file_prefix_list:
            raise ValueError(f"No prefixes found in {file_prefix_list!r}.")
        random.shuffle(self.file_prefix_list)
        self.Nsamples = len(self.file_prefix_list)

        # Created lazily in ``__getitem__`` so each DataLoader worker process
        # seeds its own generator instead of inheriting one forked copy (which
        # would make every worker draw the same time indices).
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        """Return an effectively infinite number of samples."""
        return int(1e6)

    @property
    def rng(self) -> np.random.Generator:
        """Per-process random generator, created on first use."""
        if self._rng is None:
            self._rng = np.random.default_rng()
        return self._rng

    def frame_path(self, prefix: str, file_idx: int) -> Path:
        """Return the NPZ path for one frame of a simulation.

        Args:
            prefix (str): Simulation prefix.
            file_idx (int): File (time) index.

        Returns:
            Path: Path to the frame's NPZ file.
        """
        return self.npz_dir / f"{prefix}_pvi_idx{file_idx:05d}.npz"

    def part_map(self, prefix: str) -> Mapping[str, str]:
        """Return ``{npz_part: canonical_material}`` for one simulation.

        Args:
            prefix (str): Simulation prefix.

        Returns:
            Mapping[str, str]: The part map.
        """
        raise NotImplementedError

    def _draw_triplet(self) -> tuple[int, int, int]:
        """Draw ``(prev, curr, next)`` file indices within ``[0, 99]``."""
        gap_in = int(self.rng.integers(1, self.max_timeIDX_offset, endpoint=True))
        gap_out = int(self.rng.integers(1, self.max_timeIDX_offset, endpoint=True))
        prev_idx = int(
            self.rng.integers(0, MAX_FILE_IDX - gap_in - gap_out, endpoint=True)
        )
        return prev_idx, prev_idx + gap_in, prev_idx + gap_in + gap_out

    def _load_sample(
        self, prefix: str, paths: list[Path]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Load a triplet of frames, or return ``None`` if it is unusable.

        Args:
            prefix (str): Simulation prefix.
            paths (list[Path]): ``[prev, curr, next]`` NPZ paths.

        Returns:
            tuple | None: ``(input_frames, target, lead_times, channel_map)``,
            or ``None`` if a frame lacks ``sim_time``, times do not increase,
            or no requested channel exists.
        """
        npzs = [np.load(p, allow_pickle=False) for p in paths]
        try:
            if any("sim_time" not in z.files for z in npzs):
                return None
            times = [float(z["sim_time"]) for z in npzs]
            dt_in, dt_out = times[1] - times[0], times[2] - times[1]
            if dt_in <= 0 or dt_out <= 0:
                return None

            # Only fields present in all three frames can be used.
            common = set(npzs[0].files) & set(npzs[1].files) & set(npzs[2].files)
            sources = resolve_channel_sources(
                self.part_map(prefix),
                common,
                thermo_kinds=self.thermo_kinds,
                kinematic=self.kinematic,
            )
            if not sources:
                return None

            frames = []
            for z in npzs:
                imgs = [build_channel(z, s) for s in sources]
                if not self.half_image:
                    imgs = [np.concatenate((np.fliplr(i), i), axis=1) for i in imgs]
                frames.append(torch.tensor(np.stack(imgs, axis=0), dtype=torch.float32))
        finally:
            for z in npzs:
                z.close()

        channel_map = torch.tensor(
            [canonical_index(s.canonical) for s in sources], dtype=torch.long
        )
        input_frames = torch.stack(frames[:2], dim=0)
        lead_times = torch.tensor([dt_in, dt_out], dtype=torch.float32)
        return input_frames, frames[2], lead_times, channel_map

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one ``(input_frames, target, lead_times, channel_map, id)``.

        Up to ``max_file_checks`` triplets are tried per prefix, moving to the
        next prefix (up to 5) when none are usable.

        Args:
            index (int): Sample index; wrapped onto the prefix list.

        Returns:
            tuple: The 5-tuple described in the module docstring.

        Raises:
            RuntimeError: If no usable sample is found.
        """
        index = index % self.Nsamples
        for _ in range(5):
            prefix = self.file_prefix_list[index]
            for _ in range(self.max_file_checks):
                paths = [self.frame_path(prefix, i) for i in self._draw_triplet()]
                if not all(p.is_file() for p in paths):
                    continue
                sample = self._load_sample(prefix, paths)
                if sample is not None:
                    dataset_id = torch.tensor(self.dataset_id, dtype=torch.long)
                    return (*sample, dataset_id)
            print(
                f"In {type(self).__name__}, max_file_checks reached for prefix: "
                f"{prefix}",
                file=sys.stderr,
            )
            index = (index + 1) % self.Nsamples

        raise RuntimeError(
            f"{type(self).__name__}: no usable sample found after 5 prefixes. "
            "Check npz_dir and the prefix list."
        )


class LSC_2frame_canonical_DataSet(PAGOSA_2frame_DataSet):
    """2-frame canonical-channel dataset for LSC (``lsc240420``) simulations.

    LSC fields are named by part; :data:`~yoke.datasets.pagosa_fields.
    LSC_PART_TO_MATERIAL` maps them to materials and is identical for every
    simulation.

    Args:
        LSC_NPZ_DIR (str): Directory containing LSC NPZ files.
        file_prefix_list (str): Text file listing one simulation prefix per line.
        max_timeIDX_offset (int): Maximum file-index gap between frames.
        max_file_checks (int): Triplet draws per prefix before moving on.
        half_image (bool): If False, return reflected full images.
        thermo_kinds (Iterable[str]): Thermodynamic kinds to include.
        kinematic (Iterable[str]): Kinematic fields to include.
    """

    dataset_name = "lsc"

    def __init__(
        self,
        LSC_NPZ_DIR: str,
        file_prefix_list: str,
        max_timeIDX_offset: int,
        max_file_checks: int,
        half_image: bool = True,
        thermo_kinds: Iterable[str] = ("density",),
        kinematic: Iterable[str] = KINEMATIC_FIELDS,
    ) -> None:
        """Initialize the LSC dataset."""
        super().__init__(
            LSC_NPZ_DIR,
            file_prefix_list,
            max_timeIDX_offset,
            max_file_checks,
            half_image=half_image,
            thermo_kinds=thermo_kinds,
            kinematic=kinematic,
        )

    def part_map(self, prefix: str) -> Mapping[str, str]:
        """Return the fixed LSC part map.

        Args:
            prefix (str): Simulation prefix (unused; LSC is uniform).

        Returns:
            Mapping[str, str]: :data:`LSC_PART_TO_MATERIAL`.
        """
        return LSC_PART_TO_MATERIAL


class CYL_2frame_canonical_DataSet(PAGOSA_2frame_DataSet):
    """2-frame canonical-channel dataset for Cylex (``cx241203``) simulations.

    CYL NPZ files store the ``wall``, ``maincharge`` and ``booster`` parts by
    part name and the back material by material name. The wall material comes
    from the design CSV's ``wallMat`` column, the back material from
    ``backMat``; ``maincharge`` and ``booster`` are both ``HE``. The design CSV
    is read once here (its header names and values carry leading spaces, which
    are stripped).

    Args:
        npz_dir (str): Directory containing CYL NPZ files.
        csv_filepath (str): Design CSV indexed by simulation key, with
            ``wallMat`` and ``backMat`` columns.
        file_prefix_list (str): Text file listing one simulation prefix per line.
        max_timeIDX_offset (int): Maximum file-index gap between frames.
        max_file_checks (int): Triplet draws per prefix before moving on.
        half_image (bool): If False, return reflected full images.
        thermo_kinds (Iterable[str]): Thermodynamic kinds to include.
        kinematic (Iterable[str]): Kinematic fields to include.
    """

    dataset_name = "cyl"

    def __init__(
        self,
        npz_dir: str,
        csv_filepath: str,
        file_prefix_list: str,
        max_timeIDX_offset: int,
        max_file_checks: int,
        half_image: bool = True,
        thermo_kinds: Iterable[str] = ("density",),
        kinematic: Iterable[str] = KINEMATIC_FIELDS,
    ) -> None:
        """Initialize the CYL dataset and read the design CSV."""
        super().__init__(
            npz_dir,
            file_prefix_list,
            max_timeIDX_offset,
            max_file_checks,
            half_image=half_image,
            thermo_kinds=thermo_kinds,
            kinematic=kinematic,
        )
        design = pd.read_csv(csv_filepath, index_col=0, engine="python")
        design.columns = design.columns.str.strip()
        design.index = design.index.astype(str).str.strip()
        self.materials: dict[str, tuple[str, str]] = {
            key: (str(row["wallMat"]).strip(), str(row["backMat"]).strip())
            for key, row in design[["wallMat", "backMat"]].iterrows()
        }

    def part_map(self, prefix: str) -> Mapping[str, str]:
        """Return the part map for one CYL simulation.

        The back material is stored under its own material name, so it maps to
        itself. If ``wallMat == backMat`` both parts feed one channel.

        Args:
            prefix (str): Simulation prefix (the design CSV key).

        Returns:
            Mapping[str, str]: ``{npz_part: canonical_material}``.

        Raises:
            KeyError: If ``prefix`` is not in the design CSV.
        """
        try:
            wall, back = self.materials[prefix]
        except KeyError:
            raise KeyError(f"Simulation {prefix!r} not found in design CSV.") from None
        part_map = dict(CYL_FIXED_PART_TO_MATERIAL)
        part_map["wall"] = wall
        part_map[back] = back
        return part_map
