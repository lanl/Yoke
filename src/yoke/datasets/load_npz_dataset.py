"""NPZ data loader for LodeRunner.

Functions and classes for torch `Dataset`s which sample 2D arrays from npz files that
correspond to a pre-determined list of thermodynamic and kinetic variable fields.

Currently available datasets:
- cylex (cx241203)

Authors:
Kyle Hickmann
Soumi De
Bryan Kaiser
"""

from __future__ import annotations

import contextlib
import glob
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, get_worker_info

try:
    import torch.distributed as dist  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    dist = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


def _current_rank() -> int:
    """Return current distributed rank, or 0 when unavailable."""
    try:
        if dist is not None and dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        return 0
    return 0


def rank_worker_tag(index: int | None = None) -> str:
    """Return a tag with rank/worker/pid and optional sample index.

    Args:
        index: Optional dataset index.

    Returns:
        A formatted tag string.
    """
    info = get_worker_info()
    wid = info.id if info is not None else -1
    tag = f"[rank{_current_rank()} worker{wid} pid{os.getpid()}]"
    return f"{tag} idx={index}" if index is not None else tag


def has_density_prefix(s: str) -> bool:
    """Return True if string begins with 'density_'.

    Args:
        s: Input string.

    Returns:
        True if `s` starts with "density_", else False.
    """
    return s.startswith("density_")


def extract_after_density(s: str) -> str | None:
    """Return substring after 'density_' prefix, or None if not present.

    Args:
        s: Input string.

    Returns:
        The suffix after "density_", or None.
    """
    prefix = "density_"
    if s.startswith(prefix):
        return s[len(prefix) :]
    return None


def read_npz_nan(npz: str | Path | np.lib.npyio.NpzFile, field: str) -> np.ndarray:
    """Extract a field from a .npz file and replace NaNs with 0.

    Args:
        npz: Path to a .npz file, or an opened `NpzFile` handle.
        field: Field name to extract.

    Returns:
        Field data with NaNs replaced by 0.

    Raises:
        KeyError: If the field does not exist in the NPZ.
        TypeError: If `npz` is neither a path nor an `NpzFile`.
    """
    if isinstance(npz, (str, Path)):
        npz_path = str(npz)
        with np.load(npz_path, allow_pickle=False) as data:
            if field not in data.files:
                raise KeyError(
                    f"Field {field!r} not found in {npz_path}. Available: {data.files}"
                )
            arr = data[field]
    elif isinstance(npz, np.lib.npyio.NpzFile):
        if field not in npz.files:
            raise KeyError(
                f"Field {field!r} not found in npz object. Available: {npz.files}"
            )
        arr = npz[field]
    else:
        raise TypeError(f"npz must be str/Path/NpzFile, not {type(npz)}")

    return np.nan_to_num(arr, nan=0.0)


def handle_voids(npz_filename: str | Path, hfield: str) -> np.ndarray | None:
    """Process void regions for special `_Void` hydro fields.

    If `hfield` ends with "_Void", return a mask image with NaNs where any of the
    non-void densities are present (booster/maincharge, and optionally wall). When
    `hfield` does not end with "_Void", returns None.

    Args:
        npz_filename: Path to the .npz file.
        hfield: Field name.

    Returns:
        A processed image array, or None.
    """
    if not hfield.endswith("_Void"):
        return None

    dims = np.shape(read_npz_nan(npz_filename, "av_density"))
    tmp_img = np.zeros(dims, dtype=float)

    booster = read_npz_nan(npz_filename, "density_booster")
    maincharge = read_npz_nan(npz_filename, "density_maincharge")
    mask = ~np.isnan(booster) | ~np.isnan(maincharge)

    with np.load(str(npz_filename), allow_pickle=False) as data:
        if "density_wall" in data:
            wall = read_npz_nan(data, "density_wall")
            mask |= ~np.isnan(wall)

    tmp_img[mask] = np.nan
    return tmp_img


def meshgrid_position(
    tmp_img: np.ndarray,
    npz_filename: str | Path,
    hfield: str,
) -> np.ndarray:
    """Meshgrid coordinate fields into 2D images.

    Args:
        tmp_img: The loaded 1D coordinate array.
        npz_filename: NPZ filename for fetching the companion coordinate.
        hfield: Field name ("Rcoord" or "Zcoord").

    Returns:
        A 2D meshgrid array for the requested coordinate, otherwise the original.
    """
    if hfield == "Rcoord":
        tmp_zcoord = read_npz_nan(npz_filename, "Zcoord")
        tmp_img, _ = np.meshgrid(tmp_img, tmp_zcoord)
    elif hfield == "Zcoord":
        tmp_rcoord = read_npz_nan(npz_filename, "Rcoord")
        _, tmp_img = np.meshgrid(tmp_rcoord, tmp_img)
    return tmp_img


def volfrac_density(
    tmp_img: np.ndarray,
    npz_filename: str | Path,
    hfield: str,
) -> np.ndarray:
    """Reweight densities by volume fraction for `density_*` fields.

    Args:
        tmp_img: Raw density image.
        npz_filename: NPZ file path.
        hfield: Hydro field name.

    Returns:
        Density multiplied by the corresponding volume fraction when applicable.
    """
    if not has_density_prefix(hfield):
        return tmp_img

    suffix = extract_after_density(hfield)
    if suffix is None or suffix == "":
        print(
            f"\n [load_npz_dataset.py] Could not extract suffix from hfield: {hfield!r}"
        )
        return tmp_img

    vofm_hfield = f"vofm_{suffix}"
    vofm = read_npz_nan(npz_filename, vofm_hfield)
    return tmp_img * vofm


def import_img_from_npz(npz_filename: str | Path, hfield: str) -> np.ndarray:
    """Import an image field from NPZ and apply transforms.

    Args:
        npz_filename: Path to the .npz file.
        hfield: Field name.

    Returns:
        A 2D NumPy array for the requested field.
    """
    void_img = handle_voids(npz_filename, hfield)
    tmp_img = void_img if void_img is not None else read_npz_nan(npz_filename, hfield)
    tmp_img = meshgrid_position(tmp_img, npz_filename, hfield)
    tmp_img = volfrac_density(tmp_img, npz_filename, hfield)
    return tmp_img


def combine_by_number_and_label(
    number_list: list[int],
    array: np.ndarray,
    label_list: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Combine repeated channels in a 3D array (n_list, x, z).

    Entries are combined based on repeated values in `number_list`.
    The combination fills zeros in one array with non-zeros from another.

    Args:
        number_list: Channel numbers, length n_list.
        array: Array with shape (n_list, x, z).
        label_list: Labels, length n_list.

    Returns:
        Unique channel numbers, combined array, and corresponding labels.

    Raises:
        ValueError: If inputs are inconsistent.
    """
    if len(number_list) != array.shape[0] or len(label_list) != array.shape[0]:
        raise ValueError("Mismatched input lengths.")

    number_to_indices: dict[int, list[int]] = {}
    for idx, num in enumerate(number_list):
        number_to_indices.setdefault(num, []).append(idx)

    unique_numbers = list(number_to_indices.keys())
    unique_labels: list[str] = []
    combined_arrays: list[np.ndarray] = []

    for num in unique_numbers:
        indices = number_to_indices[num]
        label = label_list[indices[0]]
        combined = np.zeros_like(array[0])
        for idx in indices:
            next_img = array[idx]
            combined = np.where(combined == 0, next_img, combined)
        combined_arrays.append(combined)
        unique_labels.append(label)

    return (np.array(unique_numbers), np.array(combined_arrays), unique_labels)


class LabeledData:
    """Relate NPZ fields to correct hydro labels via a design CSV."""

    def __init__(
        self,
        npz_filepath: str | Path,
        csv_filepath: str | Path,
        kinematic_variables: str = "velocity",
        thermodynamic_variables: str = "density",
    ) -> None:
        """Initialize labeled dataset helper.

        Args:
            npz_filepath: Path to the NPZ file.
            csv_filepath: Path to the design CSV.
            kinematic_variables: "velocity", "position", or "both".
            thermodynamic_variables: "density", "density and pressure",
                "density and energy", or "all".
        """
        self.npz_filepath = str(npz_filepath)
        self.csv_filepath = str(csv_filepath)
        self.kinematic_variables = kinematic_variables
        self.thermodynamic_variables = thermodynamic_variables

        self.key: str = ""
        self.study: str | None = None
        self.get_study_and_key(self.npz_filepath)

        self.all_hydro_field_names: list[str] = []
        self.channel_map: list[int] = []
        self.active_npz_field_names: list[str] = []
        self.active_hydro_field_names: list[str] = []

        if self.study == "cx":
            self.all_hydro_field_names = [
                "Rcoord",
                "Zcoord",
                "Uvelocity",
                "Wvelocity",
                "density_Air",
                "energy_Air",
                "pressure_Air",
                "density_Al",
                "energy_Al",
                "pressure_Al",
                "density_Be",
                "energy_Be",
                "pressure_Be",
                "density_booster",
                "energy_booster",
                "pressure_booster",
                "density_Cu",
                "energy_Cu",
                "pressure_Cu",
                "density_U.DU",
                "energy_U.DU",
                "pressure_U.DU",
                "density_maincharge",
                "energy_maincharge",
                "pressure_maincharge",
                "density_N",
                "energy_N",
                "pressure_N",
                "density_Sn",
                "energy_Sn",
                "pressure_Sn",
                "density_Steel.alloySS304L",
                "energy_Steel.alloySS304L",
                "pressure_Steel.alloySS304L",
                "density_Polymer.Sylgard",
                "energy_Polymer.Sylgard",
                "pressure_Polymer.Sylgard",
                "density_Ta",
                "energy_Ta",
                "pressure_Ta",
                "density_Void",
                "energy_Void",
                "pressure_Void",
                "density_Water",
                "energy_Water",
                "pressure_Water",
            ]
            self.channel_map = list(range(len(self.all_hydro_field_names)))
            self.cylex_data_loader()
        else:
            raise ValueError(
                "Hydro field information unavailable for specified dataset/study."
            )

    def get_active_hydro_indices(self) -> list[int]:
        """Return indices of active hydro fields within the full list."""
        return [
            self.all_hydro_field_names.index(field)
            for field in self.active_hydro_field_names
            if field in self.all_hydro_field_names
        ]

    def cylex_data_loader(self) -> None:
        """Configure active fields and channel map for the cylex dataset."""
        design_df = pd.read_csv(
            self.csv_filepath,
            sep=",",
            header=0,
            index_col=0,
            engine="python",
        )
        for col in design_df.columns:
            design_df.rename(columns={col: col.strip()}, inplace=True)

        non_he_mats_arr = design_df.loc[self.key, "wallMat":"backMat"].values
        non_he_mats = [str(m).strip() for m in non_he_mats_arr]

        self.channel_map = []
        self.active_npz_field_names = []
        self.active_hydro_field_names = []

        if self.kinematic_variables == "velocity":
            self.active_hydro_field_names = ["Uvelocity", "Wvelocity"]
            self.active_npz_field_names = list(self.active_hydro_field_names)
        elif self.kinematic_variables == "position":
            self.active_hydro_field_names = ["Rcoord", "Zcoord"]
            self.active_npz_field_names = list(self.active_hydro_field_names)
        elif self.kinematic_variables == "both":
            self.active_hydro_field_names = [
                "Rcoord",
                "Zcoord",
                "Uvelocity",
                "Wvelocity",
            ]
            self.active_npz_field_names = list(self.active_hydro_field_names)
        else:
            raise ValueError(
                "Incorrectly specified kinematic_variables. Choose from 'velocity', "
                "'position', or 'both'."
            )

        self.active_npz_field_names.extend(["density_wall", f"density_{non_he_mats[1]}"])
        self.active_hydro_field_names.extend(
            [f"density_{non_he_mats[0]}", f"density_{non_he_mats[1]}"]
        )
        self.active_npz_field_names.extend(["density_maincharge", "density_booster"])
        self.active_hydro_field_names.extend(["density_maincharge", "density_booster"])

        if self.thermodynamic_variables in ("density and pressure", "all"):
            self.active_npz_field_names.extend(
                ["pressure_wall", f"pressure_{non_he_mats[1]}"]
            )
            self.active_hydro_field_names.extend(
                [f"pressure_{non_he_mats[0]}", f"pressure_{non_he_mats[1]}"]
            )
            self.active_npz_field_names.extend(
                ["pressure_maincharge", "pressure_booster"]
            )
            self.active_hydro_field_names.extend(
                ["pressure_maincharge", "pressure_booster"]
            )
        elif self.thermodynamic_variables in ("density and energy", "all"):
            self.active_npz_field_names.extend(
                ["energy_wall", f"energy_{non_he_mats[1]}"]
            )
            self.active_hydro_field_names.extend(
                [f"energy_{non_he_mats[0]}", f"energy_{non_he_mats[1]}"]
            )
            self.active_npz_field_names.extend(["energy_maincharge", "energy_booster"])
            self.active_hydro_field_names.extend(["energy_maincharge", "energy_booster"])
        elif self.thermodynamic_variables != "density":
            raise ValueError(
                "Incorrectly specified thermodynamic_variables. Choose from 'density', "
                "'density and pressure', 'density and energy', or 'all'."
            )

        self.channel_map = self.get_active_hydro_indices()

    @staticmethod
    def extract_letters(s: str) -> str | None:
        """Match letters at the beginning until the first digit."""
        match = re.match(r"([a-zA-Z]+)\d", s)
        return match.group(1) if match else None

    def get_study_and_key(self, npz_filepath: str) -> None:
        """Extract simulation key and study from an NPZ filename."""
        self.key = npz_filepath.split("/")[-1].split("_pvi_")[0]
        self.study = self.extract_letters(self.key)

    def get_all_hydro_field_names(self) -> list[str]:
        """Return all possible hydro field names."""
        return self.all_hydro_field_names

    def get_channel_map(self) -> list[int]:
        """Return the channel map (active channel indices)."""
        return list(self.channel_map)

    def get_active_hydro_field_names(self) -> list[str]:
        """Return only the active hydro field names."""
        return list(self.active_hydro_field_names)

    def get_active_npz_field_names(self) -> list[str]:
        """Return only the active field names in an NPZ file."""
        return list(self.active_npz_field_names)


def process_channel_data(
    channel_map: list[int],
    img_list_combined: np.ndarray,
    active_hydro_field_names: list[str],
) -> tuple[list[int], np.ndarray, list[str]]:
    """Make channel entries unique by combining repeated channels.

    Args:
        channel_map: Channel indices.
        img_list_combined: Array with shape (n_t, n_c, x, z) in list form.
        active_hydro_field_names: Names for each channel.

    Returns:
        Updated channel map, updated images, updated names.
    """
    unique_channels = np.unique(channel_map)
    if len(unique_channels) >= len(channel_map):
        return (channel_map, img_list_combined, active_hydro_field_names)

    new_img_list_combined: list[np.ndarray] = []
    new_channel_map: list[np.ndarray] = []
    new_active_names: list[list[str]] = []

    for i in np.arange(img_list_combined.shape[0]):
        channel_map_u, new_img, names_u = combine_by_number_and_label(
            channel_map,
            img_list_combined[i],
            active_hydro_field_names,
        )
        new_img_list_combined.append(new_img)
        new_channel_map.append(channel_map_u)
        new_active_names.append(names_u)

    img_list_combined_new = np.array(new_img_list_combined)
    if len(np.unique(new_channel_map[0])) < len(new_channel_map[0]):
        raise RuntimeError("Combination of repeated materials failed.")

    channel_map_list = new_channel_map[0].astype(int).tolist()
    return (channel_map_list, img_list_combined_new, new_active_names[0])


_TemporalSample = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


class TemporalDataSet(Dataset[_TemporalSample]):
    """Temporal field-to-field mapping dataset.

    Returns:
        (start_img, channel_map, end_img, channel_map, Dt)
    """

    def __init__(
        self,
        npz_dir: str,
        csv_filepath: str,
        file_prefix_list: str,
        max_timeIDX_offset: int,
        max_file_checks: int,
        half_image: bool = True,
        thermodynamic_variables: str = "density",
        kinematic_variables: str = "velocity",
    ) -> None:
        """Initialize TemporalDataSet.

        This dataset returns multi-channel images at two different times from
        the *Cylex* simulation. The *maximum time-offset* can be specified. The channels
        in the images returned are the kinematic (position, velocity) and thermodynamic
        (pressure, density, energy) fields for each material as requested by the user.
        The time-offset between the two images is also returned.

        Args:
            npz_dir (str): Directory path for CYL NPZ files.
            csv_filepath: Full path for the design CSV.
            file_prefix_list (str): Text file listing unique prefixes corresponding
                                    to unique simulations.
            max_timeIDX_offset (int): Maximum timesteps-ahead to attempt
                                      prediction for. A prediction image will be chosen
                                      within this timeframe at random.
            max_file_checks (int): This dataset generates two random time indices and
                                   checks if the corresponding files exist. This
                                   argument controls the maximum number of times indices
                                   are generated before throwing an error.
            half_image (bool): If True then returned images are NOT reflected about axis
                               of symmetry and half-images are returned instead.
            kinematic_variables (str): "velocity", "position", or "both".
            thermodynamic_variables (str): "density", "density and pressure",
                                           "density and energy", or "all".

        """
        self.npz_dir = npz_dir
        self.csv_filepath = csv_filepath
        self.max_timeIDX_offset = max_timeIDX_offset
        self.max_file_checks = max_file_checks
        self.half_image = half_image
        self.thermodynamic_variables = thermodynamic_variables
        self.kinematic_variables = kinematic_variables

        with open(file_prefix_list, encoding="utf-8") as f:
            self.file_prefix_list = [line.rstrip() for line in f]

        random.shuffle(self.file_prefix_list)
        self.n_samples = len(self.file_prefix_list)
        self.rng = np.random.default_rng()
        self._build_valid_prefixes()

        self.channel_map: list[int] = []
        self.active_npz_field_names: list[str] = []
        self.active_hydro_field_names: list[str] = []
        self.all_hydro_field_names: list[str] = []

    def __len__(self) -> int:
        """Return an effectively infinite number of samples for training."""
        return 800_000

    def __getitem__(self, index: int) -> _TemporalSample:
        """Return one temporal training sample."""
        index = index % self.n_samples
        depth = int(getattr(self, "_fallback_depth", 0))

        prefix_attempt = 0
        while prefix_attempt < 5:
            file_prefix = self.file_prefix_list[index]
            attempt = 0

            while attempt < self.max_file_checks:
                seq_len = int(
                    self.rng.integers(0, self.max_timeIDX_offset, endpoint=True)
                )
                start_idx = int(self.rng.integers(0, 100 - seq_len, endpoint=True))
                end_idx = start_idx + seq_len

                start_file = f"{file_prefix}_pvi_idx{start_idx:05d}.npz"
                end_file = f"{file_prefix}_pvi_idx{end_idx:05d}.npz"

                start_file_path = Path(self.npz_dir) / start_file
                end_file_path = Path(self.npz_dir) / end_file

                if not (start_file_path.is_file() and end_file_path.is_file()):
                    attempt += 1
                    continue

                try:
                    start_npz = np.load(str(start_file_path), allow_pickle=False)
                except OSError:
                    attempt += 1
                    continue

                try:
                    end_npz = np.load(str(end_file_path), allow_pickle=False)
                except OSError:
                    start_npz.close()
                    attempt += 1
                    continue

                try:
                    ld = LabeledData(
                        str(start_file_path),
                        self.csv_filepath,
                        thermodynamic_variables=self.thermodynamic_variables,
                        kinematic_variables=self.kinematic_variables,
                    )
                    active_npz_field_names = ld.get_active_npz_field_names()
                    active_hydro_field_names = ld.get_active_hydro_field_names()
                    channel_map = ld.get_channel_map()
                    self.all_hydro_field_names = ld.get_all_hydro_field_names()

                    available_start = set(start_npz.files)
                    mask_start = [f in available_start for f in active_npz_field_names]
                    fields_start = [
                        f for f, keep in zip(active_npz_field_names, mask_start) if keep
                    ]
                    chmap_start = [
                        cm for cm, keep in zip(channel_map, mask_start) if keep
                    ]
                    names_start = [
                        nm
                        for nm, keep in zip(active_hydro_field_names, mask_start)
                        if keep
                    ]
                    if not fields_start:
                        start_npz.close()
                        end_npz.close()
                        attempt += 1
                        continue

                    available_end = set(end_npz.files)
                    mask_both = [f in available_end for f in fields_start]

                    present_fields = [
                        f for f, keep in zip(fields_start, mask_both) if keep
                    ]
                    filtered_chmap = [
                        cm for cm, keep in zip(chmap_start, mask_both) if keep
                    ]
                    filtered_names = [
                        nm for nm, keep in zip(names_start, mask_both) if keep
                    ]
                    if not present_fields:
                        start_npz.close()
                        end_npz.close()
                        attempt += 1
                        continue

                    self.active_npz_field_names = present_fields
                    self.channel_map = filtered_chmap
                    self.active_hydro_field_names = filtered_names

                    start_img_list: list[np.ndarray] = []
                    end_img_list: list[np.ndarray] = []

                    for hfield in present_fields:
                        tmp = import_img_from_npz(start_file_path, hfield)
                        if not self.half_image:
                            tmp = np.concatenate((np.fliplr(tmp), tmp), axis=1)
                        start_img_list.append(tmp)

                    for hfield in present_fields:
                        tmp = import_img_from_npz(end_file_path, hfield)
                        if not self.half_image:
                            tmp = np.concatenate((np.fliplr(tmp), tmp), axis=1)
                        end_img_list.append(tmp)

                    img_list_combined = np.array([start_img_list, end_img_list])
                    channel_map_u, img_list_combined, names_u = process_channel_data(
                        self.channel_map,
                        img_list_combined,
                        self.active_hydro_field_names,
                    )
                    self.channel_map = channel_map_u
                    self.active_hydro_field_names = names_u

                    start_img = torch.as_tensor(
                        np.stack(img_list_combined[0], axis=0),
                        dtype=torch.float32,
                    ).contiguous()
                    end_img = torch.as_tensor(
                        np.stack(img_list_combined[1], axis=0),
                        dtype=torch.float32,
                    ).contiguous()

                    dt = torch.tensor(0.25 * (end_idx - start_idx), dtype=torch.float32)
                    cm_tensor = torch.as_tensor(self.channel_map, dtype=torch.long)

                    start_npz.close()
                    end_npz.close()
                    return (start_img, cm_tensor, end_img, cm_tensor, dt)
                except Exception:
                    with contextlib.suppress(Exception):
                        start_npz.close()
                    with contextlib.suppress(Exception):
                        end_npz.close()
                    attempt += 1
                    continue

            print(
                f"In TemporalDataSet, max_file_checks reached for prefix: {file_prefix}",
                file=sys.stderr,
            )
            prefix_attempt += 1
            index = (index + 1) % self.n_samples

        if depth < 10:
            print(
                "[TemporalDataSet] WARN: skipping index after multiple failures; "
                f"index={index}",
                file=sys.stderr,
            )
            self._fallback_depth = depth + 1
            try:
                return self.__getitem__((index + 1) % self.n_samples)
            finally:
                self._fallback_depth = depth

        for _ in range(50):
            j = int(self.rng.integers(0, self.n_samples))
            try:
                return self.__getitem__(j)
            except Exception:
                continue

        raise RuntimeError(
            "TemporalDataSet: unable to assemble any sample after global retries. "
            "Check npz_dir / CSV alignment or loosen selection rules."
        )

    def _probe_prefix_once(self, prefix: str) -> dict[str, Any] | None:
        """Probe a prefix by trying a handful of time indices."""
        candidates = [0, 10, 20, 40, 60, 80, 99]
        rng = np.random.default_rng(123)
        rng.shuffle(candidates)

        for start_idx in candidates[:5]:
            start_file = f"{prefix}_pvi_idx{start_idx:05d}.npz"
            start_fp = Path(self.npz_dir) / start_file
            if not start_fp.is_file():
                continue

            try:
                with np.load(str(start_fp), allow_pickle=False) as z:
                    ld = LabeledData(
                        str(start_fp),
                        self.csv_filepath,
                        thermodynamic_variables=self.thermodynamic_variables,
                        kinematic_variables=self.kinematic_variables,
                    )
                    active_fields = ld.get_active_npz_field_names()
                    available = set(z.files)
                    present_fields = [f for f in active_fields if f in available]
                    if present_fields:
                        return {"prefix": prefix, "present_fields": present_fields}
            except OSError:
                continue

        return None

    def _build_valid_prefixes(self) -> None:
        """Index prefixes that look usable by probing one NPZ each."""
        prefixes = list(getattr(self, "file_prefix_list", []))
        if not prefixes:
            files = glob.glob(str(Path(self.npz_dir) / "*_pvi_idx*.npz"))
            rx = re.compile(r"_pvi_idx\d+\.npz$")
            prefixes = sorted({rx.sub("", os.path.basename(f)) for f in files})
            self.file_prefix_list = prefixes

        total = len(prefixes)
        hits: list[dict[str, Any]] = []

        for p in prefixes:
            h = self._probe_prefix_once(p)
            if h is not None:
                hits.append(h)

        if hits:
            self.valid_prefixes = np.array([h["prefix"] for h in hits])
            self._probe_present_fields = {h["prefix"]: h["present_fields"] for h in hits}
            self.n_valid = len(self.valid_prefixes)
            print(
                f"[TemporalDataSet] Indexed {self.n_valid}/{total} prefixes via probe.",
                file=sys.stderr,
            )
        else:
            self.valid_prefixes = np.array(prefixes)
            self.n_valid = len(self.valid_prefixes)
            print(
                "[TemporalDataSet] WARN: probe found 0 usable prefixes; falling back to "
                f"all {self.n_valid} prefixes. Bad samples will be skipped in "
                "__getitem__.",
                file=sys.stderr,
            )


_SequentialSample = tuple[torch.Tensor, torch.Tensor, list[int]]


class SequentialDataSet(Dataset[_SequentialSample]):
    """Return a sequence of consecutive frames from a Cylex simulation.

    This dataset returns sequences of frames of Cylex simulation data at specified
    time offsets and sequence lengths.  For a given sequence length, multiple
    time offsets are allowed.  For example, if seq_len=2 and timeIDX_offset=[1, 2],
    this dataset will contain all Cylex simulation sequences of length 2 with frames
    offset by 1 and 2 time indices (i.e., the set of sequences
    (t, t+1), (t, t+2), (t+1, t+2), (t+1, t+3), ...).

    Args:
        npz_dir (str): Directory path for CYL NPZ files.
        csv_filepath: Full path for the design CSV.
        file_prefix_list (str): Text file listing unique prefixes corresponding
                                to unique simulations.
        max_file_checks (int): This dataset generates two random time indices and
                               checks if the corresponding files exist. This
                               argument controls the maximum number of times indices
                               are generated before throwing an error.
        seq_len (int): Number of consecutive frames to return. This includes the
                       starting frame.
        half_image (bool): If True then returned images are NOT reflected about axis
                           of symmetry and half-images are returned instead.
        kinematic_variables (str): "velocity", "position", or "both".
        thermodynamic_variables (str): "density", "density and pressure",
                                       "density and energy", or "all".

    """

    def __init__(
        self,
        npz_dir: str,
        csv_filepath: str,
        file_prefix_list: str,
        max_file_checks: int,
        seq_len: int,
        half_image: bool = True,
        kinematic_variables: str = "velocity",
        thermodynamic_variables: str = "density",
    ) -> None:
        """Initialize SequentialDataSet."""
        dir_path = Path(npz_dir)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {npz_dir}")

        self.npz_dir = npz_dir
        self.csv_filepath = csv_filepath
        self.max_file_checks = max_file_checks
        self.seq_len = seq_len
        self.half_image = half_image
        self.thermodynamic_variables = thermodynamic_variables
        self.kinematic_variables = kinematic_variables

        with open(file_prefix_list, encoding="utf-8") as f:
            self.file_prefix_list = [line.rstrip() for line in f]

        random.shuffle(self.file_prefix_list)
        self.n_samples = len(self.file_prefix_list)
        self.rng = np.random.default_rng()

        self.channel_map: list[int] = []
        self.active_npz_field_names: list[str] = []
        self.active_hydro_field_names: list[str] = []

    def __len__(self) -> int:
        """Return number of unique simulation prefixes."""
        return self.n_samples

    def __getitem__(self, index: int) -> _SequentialSample:
        """Return one sequence sample: (img_seq, dt, channel_map)."""
        index = index % self.n_samples
        file_prefix = self.file_prefix_list[index]

        prefix_attempt = 0
        file_paths: list[Path] = []

        while prefix_attempt < self.max_file_checks:
            start_idx = int(self.rng.integers(0, 100 - self.seq_len))
            valid_sequence = True
            file_paths = []

            for offset in range(self.seq_len):
                idx = start_idx + offset
                file_name = f"{file_prefix}_pvi_idx{idx:05d}.npz"
                file_path = Path(self.npz_dir) / file_name
                if not file_path.is_file():
                    valid_sequence = False
                    break
                file_paths.append(file_path)

            if valid_sequence:
                break

            prefix_attempt += 1
            index = (index + 1) % self.n_samples

        if prefix_attempt == self.max_file_checks:
            raise RuntimeError(
                f"Failed to find valid sequence for prefix: {file_prefix} after "
                f"{self.max_file_checks} attempts."
            )

        frames: list[torch.Tensor] = []
        channel_map: list[int] = []
        active_hydro_field_names: list[str] = []

        for file_path in file_paths:
            try:
                data_npz = np.load(str(file_path), allow_pickle=False)
                ld = LabeledData(
                    str(file_path),
                    self.csv_filepath,
                    thermodynamic_variables=self.thermodynamic_variables,
                    kinematic_variables=self.kinematic_variables,
                )
                self.active_npz_field_names = ld.get_active_npz_field_names()
                active_hydro_field_names = ld.get_active_hydro_field_names()
                channel_map = ld.get_channel_map()

                field_imgs: list[np.ndarray] = []
                for hfield in self.active_npz_field_names:
                    tmp_img = import_img_from_npz(file_path, hfield)
                    if not self.half_image:
                        tmp_img = np.concatenate((np.fliplr(tmp_img), tmp_img), axis=1)
                    field_imgs.append(tmp_img)

                data_npz.close()

                img_list_combined = np.array([field_imgs])
                channel_map_u, img_list_combined, names_u = process_channel_data(
                    channel_map,
                    img_list_combined,
                    active_hydro_field_names,
                )
                channel_map = channel_map_u
                active_hydro_field_names = names_u

                field_tensor = torch.as_tensor(
                    np.stack(img_list_combined[0], axis=0),
                    dtype=torch.float32,
                ).contiguous()
                frames.append(field_tensor)
            except OSError as exc:
                raise RuntimeError(f"Error loading file: {file_path}") from exc

        self.channel_map = channel_map
        self.active_hydro_field_names = active_hydro_field_names

        img_seq = torch.stack(frames, dim=0).contiguous()
        dt = torch.tensor(0.25, dtype=torch.float32)

        return (img_seq, dt, list(self.channel_map))
