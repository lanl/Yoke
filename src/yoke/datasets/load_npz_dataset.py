"""Npz data loader for loderunner.

Functions and classes for torch DataSets which sample 2D arrays from npz files
that corresponded to a pre-determined list of thermodynamic and kinetic variable
fields.

Currently available datasets:
- cylex (cx241203)

Authors:
Kyle Hickmann
Soumi De
Bryan Kaiser

"""

####################################
# Packages
####################################
import sys
from pathlib import Path
import typing
import random
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

NoneStr = typing.Union[None, str]


def import_img_from_npz(npz_filename: str, hfield: str) -> np.ndarray:
    """Imports image data from npz file."""
    tmp_img = read_npz_nan(npz_filename, hfield)
    # If hfield = Rcoord or Zcoord, then meshgrid:
    tmp_img = meshgrid_position(tmp_img, npz_filename, hfield)
    # If hfield = density_..., then multiply by volume fraction:
    tmp_img = volfrac_density(tmp_img, npz_filename, hfield)
    return tmp_img


def volfrac_density(tmp_img: np.ndarray, npz_filename: str, hfield: str) -> np.ndarray:
    """Reweigh densities by volume fraction.

    If `hfield` has the prefix 'density_', multiply `tmp_img` by the corresponding
    volume fraction field from the .npz file.
    """
    if not has_density_prefix(hfield):
        return tmp_img
    suffix = extract_after_density(hfield)
    if not suffix:
        print(
            f"\n [load_npz_dataset.py] Could not extractsuffix from hfield: '{hfield}'"
        )
        return tmp_img
    vofm_hfield = "vofm_" + suffix
    vofm = read_npz_nan(npz_filename, vofm_hfield)
    return tmp_img * vofm


def meshgrid_position(tmp_img: np.ndarray, npz_filename: str, hfield: str) -> np.ndarray:
    """If hfield = position, then meshgrid the arrays."""
    if hfield == "Rcoord":
        tmp_zcoord = read_npz_nan(npz_filename, "Zcoord")
        tmp_img, _ = np.meshgrid(tmp_img, tmp_zcoord)
    elif hfield == "Zcoord":
        tmp_rcoord = read_npz_nan(npz_filename, "Rcoord")
        _, tmp_img = np.meshgrid(tmp_rcoord, tmp_img)
    return tmp_img


def extract_after_density(s: str) -> None:
    """Get the name of the material."""
    prefix = "density_"
    if s.startswith(prefix):
        return s[len(prefix) :]
    return None


def has_density_prefix(s: str) -> None:
    """Returns True if string begins with 'density'."""
    return s.startswith("density_")


def combine_by_number_and_label(
    number_list: list[int], array: np.ndarray, label_list: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Combine entries in a 3D array (n_list, x, z).

    Entries are combined based on repeated values in number_list and
    label_list. The combination fills 0s in one array with
    non-0s from another. If both are non-zero, values are summed.

    Args:
        number_list: List of integers (length n_list).
        array: 3D NumPy array of shape (n_list, x, z).
        label_list: List of strings (length n_list).

    Returns:
        unique_numbers: Array of unique channel numbers.
        combined_array: Array of shape (n_unique, x, z).
        unique_labels: Corresponding hydrofield labels for unique channels.
    """
    assert len(number_list) == array.shape[0] == len(label_list), (
        "Mismatched input lengths."
    )

    number_to_index = {}
    for idx, num in enumerate(number_list):
        if num not in number_to_index:
            number_to_index[num] = [idx]
        else:
            number_to_index[num].append(idx)

    unique_numbers = list(number_to_index.keys())
    unique_labels = []
    combined_arrays = []

    for num in unique_numbers:
        indices = number_to_index[num]
        combined = array[indices[0]].copy()
        label = label_list[indices[0]]

        combined = np.zeros_like(array[0])

        for idx in indices:
            next_img = array[idx]
            combined = np.where(combined == 0, next_img, combined)

        combined_arrays.append(combined)
        unique_labels.append(label)

    return np.array(unique_numbers), np.array(combined_arrays), unique_labels


def read_npz_nan(npz: np.lib.npyio.NpzFile, field: str) -> np.ndarray:
    """Extract a specific field from a .npz file and replace NaNs with 0.

    Args:
        npz (np.lib.npyio.NpzFile): Loaded .npz file.
        field (str): Field name to extract.

    Returns:
        np.ndarray: Field data with NaNs replaced by 0.

    """
    return np.nan_to_num(npz[field], nan=0.0)


class LabeledData:
    """A class to process datasets by relating input data to correct labels.

    Use this to get correctly labeled hydro fields and channel maps.
    """

    def __init__(
        self,
        npz_filepath: str,
        csv_filepath: str,
        kinematic_variables: str = "velocity",
        thermodynamic_variables: str = "density",
    ) -> None:
        """Initializes the dataset processor.

        Parameters:
        - npz_filepath (str): Path to the hydro field data file (NPZ).
        - csv_filepath (str): Path to the 'design' file (CSV).
        """
        self.npz_filepath = npz_filepath
        self.csv_filepath = csv_filepath
        self.kinematic_variables = kinematic_variables
        self.thermodynamic_variables = thermodynamic_variables

        # Get the hydro_fields
        self.get_study_and_key(self.npz_filepath)
        if self.study == "cx":  # cylex dataset
            self.all_hydro_field_names = [
                "Rcoord",  # 4 kinematic variable fields
                "Zcoord",
                "Uvelocity",
                "Wvelocity",
                "density_Air",  # 39 thermodynamic variable fields
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

            # # channel_map (later in_vars) integer labels for each field
            self.channel_map = np.arange(0, len(self.all_hydro_field_names))

            # get the material names that are present in this npz file
            self.cylex_data_loader()

        else:
            print(
                "\n ERROR: hydro_field information unavailable"
                "for specified dataset. -> See load_npz_dataset.py\n"
            )

    def get_active_hydro_indices(self) -> list:
        """Returns the indices of active_hydro_field_names within hydro_field_names."""
        return [
            self.all_hydro_field_names.index(field)
            for field in self.active_hydro_field_names
            if field in self.all_hydro_field_names
        ]

    def cylex_data_loader(self) -> None:
        """Data loader for the cylex dataset.

        Pairs the data arrays in the .npz file with the corresponding elements of
        hydro_field_names by using the columns in the .csv design file.
        """
        design_df = pd.read_csv(
            self.csv_filepath, sep=",", header=0, index_col=0, engine="python"
        )

        # removed spaces from headers:
        for col in design_df.columns:
            design_df.rename(columns={col: col.strip()}, inplace=True)

        # get the names of the non-HE material(s) from the design file:
        non_he_mats = design_df.loc[self.key, "wallMat":"backMat"].values
        non_he_mats = [
            m.strip() for m in non_he_mats
        ]  # materials that are not the high explosive material

        # get the hydro_field_names and the corresponding channel indices
        # for the given npz data:
        self.channel_map = []
        self.active_npz_field_names = []
        self.active_hydro_field_names = []

        if self.kinematic_variables == "velocity":
            self.active_hydro_field_names = ["Uvelocity", "Wvelocity"]
            self.active_npz_field_names = self.active_hydro_field_names
        elif self.kinematic_variables == "position":
            self.active_hydro_field_names = ["Rcoord", "Zcoord"]
            self.active_npz_field_names = self.active_hydro_field_names
        elif self.kinematic_variables == "both":
            self.active_hydro_field_names = [
                "Rcoord",
                "Zcoord",
                "Uvelocity",
                "Wvelocity",
            ]
            self.active_npz_field_names = self.active_hydro_field_names
        else:
            raise ValueError(
                "\n ERROR: Failure to load data. "
                "Incorrectly specified kinematic "
                "variables: Choose from 'velocity'"
                "(default), 'position', or 'both'."
            )

        # Note: wall and background materials (specific to cylex):
        # In the csv file the materials are called as `wall_mat' and
        # `back_mat' but in the npz files, back_mat is the material name.

        # Wall & background material densities:
        self.active_npz_field_names = np.append(
            self.active_npz_field_names,
            ["density_wall", "density_" + non_he_mats[1]],
        )
        self.active_hydro_field_names = np.append(
            self.active_hydro_field_names,
            ["density_" + non_he_mats[0], "density_" + non_he_mats[1]],
        )
        # HE density:
        self.active_npz_field_names = np.append(
            self.active_npz_field_names, ["density_maincharge"]
        )
        self.active_hydro_field_names = np.append(
            self.active_hydro_field_names, ["density_maincharge"]
        )
        self.active_npz_field_names = np.append(
            self.active_npz_field_names, ["density_booster"]
        )
        self.active_hydro_field_names = np.append(
            self.active_hydro_field_names, ["density_booster"]
        )

        if self.thermodynamic_variables in ("density and pressure", "all"):
            # Wall & background material pressures:
            self.active_npz_field_names = np.append(
                self.active_npz_field_names,
                ["pressure_wall", "pressure_" + non_he_mats[1]],
            )
            self.active_hydro_field_names = np.append(
                self.active_hydro_field_names,
                ["pressure_" + non_he_mats[0], "pressure_" + non_he_mats[1]],
            )
            # HE pressures:
            self.active_npz_field_names = np.append(
                self.active_npz_field_names, ["pressure_maincharge"]
            )
            self.active_hydro_field_names = np.append(
                self.active_hydro_field_names, ["pressure_maincharge"]
            )
            self.active_npz_field_names = np.append(
                self.active_npz_field_names, ["pressure_booster"]
            )
            self.active_hydro_field_names = np.append(
                self.active_hydro_field_names, ["pressure_booster"]
            )

        elif self.thermodynamic_variables in ("density and energy", "all"):
            # Wall & background material energy:
            self.active_npz_field_names = np.append(
                self.active_npz_field_names,
                ["energy_wall", "energy_" + non_he_mats[1]],
            )
            self.active_hydro_field_names = np.append(
                self.active_hydro_field_names,
                ["energy_" + non_he_mats[0], "energy_" + non_he_mats[1]],
            )
            # HE energy:
            self.active_npz_field_names = np.append(
                self.active_npz_field_names, ["energy_maincharge"]
            )
            self.active_hydro_field_names = np.append(
                self.active_hydro_field_names, ["energy_maincharge"]
            )
            self.active_npz_field_names = np.append(
                self.active_npz_field_names, ["energy_booster"]
            )
            self.active_hydro_field_names = np.append(
                self.active_hydro_field_names, ["energy_booster"]
            )

        elif self.thermodynamic_variables != "density":
            raise ValueError(
                "\n ERROR: Failure to load data. "
                "Incorrectly specified thermodynamic "
                "variables: Choose from 'density'"
                "(default), 'density and pressure', "
                "'density and energy', or 'all'."
            )

        self.channel_map = self.get_active_hydro_indices()

    def extract_letters(self, s: str) -> str:
        """Match letters at the beginning until the first digit."""
        match = re.match(r"([a-zA-Z]+)\d", s)
        return match.group(1) if match else None

    def get_study_and_key(self, npz_filepath: str) -> str:
        """Simulation key extraction.

        Function to extract simulation *key* from the name of an .npz file.

        A study key looks like **lsc240420_id00001** and a NPZ filename is like
        **lsc240420_id00001_pvi_idx00000.npz**

        Args:
            npz_filepath (str): file path from working directory to .npz file

        Returns:
            key (str): The corresponding simulation key for the NPZ file.
                       E.g., 'cx241203_id01250'
            study (str): The name of the study/dataset. E.g., 'cx'.

        """
        self.key = npz_filepath.split("/")[-1].split("_pvi_")[0]
        self.study = self.extract_letters(self.key)

    def get_hydro_field_names(self) -> list[str]:
        """Returns all possible hydro field names."""
        return self.all_hydro_field_names

    def get_channel_map(self) -> list[str]:
        """Returns channel_map (a vector of active channel numbers)."""
        return self.channel_map

    def get_active_hydro_field_names(self) -> list[str]:
        """Returns only the active hydro field names."""
        return self.active_hydro_field_names

    def get_active_npz_field_names(self) -> list[str]:
        """Returns only the active field names in a npz file."""
        return self.active_npz_field_names


def process_channel_data(
    channel_map: list, img_list_combined: np.ndarray, active_hydro_field_names: list
) -> tuple[list, np.ndarray, list]:
    """Processes channel data so that they are unique entries.

    Given a channel map, combined image lists, and active hydro field names,
    returns a channel map with unique values and the corresponding combined
    image list and active hydro field names.

    Args:
        channel_map (list): list of indices of active channels (fields).
        img_list_combined (array): Numpy array combining multiple image lists
                                 where each image list is a list of images
                                 for all hydro fields in a simulation.
        active_hydro_field_names (list): list of active hydro fields.

    Returns:
        channel_map (list): Unique channels.
        img_list_combined (array): Combined image lists corresponding to the
                                 unique channels.
        active_hydro_field_names (list): list of active hydro fields corresponding
                                       to the unique channels.
    """
    unique_channels = np.unique(channel_map)
    if len(unique_channels) < len(channel_map):
        # Assumes size(img_list_combined) = [n_images,n_hydro_fields,n_x,n_z]
        new_img_list_combined = []
        new_channel_map = []
        new_active_hydro_field_names = []
        for i in np.arange(img_list_combined.shape[0]):
            result = combine_by_number_and_label(
                channel_map,
                img_list_combined[i],
                active_hydro_field_names,
            )
            channel_map_, new_img, active_hydro_field_names_ = result
            new_img_list_combined.append(new_img)
            new_channel_map.append(channel_map_)
            new_active_hydro_field_names.append(active_hydro_field_names_)
        img_list_combined = np.array(new_img_list_combined)
        if len(np.unique(new_channel_map[0])) < len(new_channel_map[0]):
            print("\n ERROR: combination of repeated materials fail")
    return new_channel_map[0], img_list_combined, new_active_hydro_field_names[0]


class TemporalDataSet(Dataset):
    """Temporal field-to-field mapping dataset.

    Maps hydrofield .npz data to correct material labels in .csv 'design' file.
    This dataset returns multi-channel images at two different times from a
    simulation. The *maximum time-offset* can be specified. The channels in the
    images returned are the densities for each material at a given time as well
    as the (R, Z)-velocity fields. The time-offset between the two images is
    also returned.

    NOTE: The way time indices are chosen necessitates *max_time_idx_offset*
    being less than or equal to 3 in the lsc240420 data.

    Args:
        npz_dir (str): Directory storing NPZ files of the dataset being analyzed.
        csv_filepath (str): Path to the 'design' file (CSV).
        file_prefix_list (str): Text file listing unique prefixes corresponding
                                to unique simulations.
        max_time_idx_offset (int): Maximum timesteps-ahead to attempt
                                prediction for. A prediction image will be chosen
                                within this timeframe at random.
        max_file_checks (int): This dataset generates two random time indices and
                                checks if the corresponding files exist. This
                                argument controls the maximum number of times indices
                                are generated before throwing an error.
        half_image (bool): If True then returned images are NOT reflected about axis
                                of symmetry and half-images are returned instead.
    """

    def __init__(
        self,
        npz_dir: str,
        csv_filepath: str,
        file_prefix_list: str,
        max_time_idx_offset: int,
        max_file_checks: int,
        half_image: bool = True,
    ) -> None:
        """Initialization of timestep dataset."""
        self.npz_dir = npz_dir
        self.csv_filepath = csv_filepath
        self.max_time_idx_offset = max_time_idx_offset
        self.max_file_checks = max_file_checks
        self.half_image = half_image

        # Create filelist
        with open(file_prefix_list, encoding="utf-8") as f:
            self.file_prefix_list = [line.rstrip() for line in f]

        # Shuffle the list of prefixes in-place
        random.shuffle(self.file_prefix_list)
        self.n_samples = len(self.file_prefix_list)

        # Initialize random number generator for time index selection
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        """Return effectively infinite number of samples in dataset."""
        return int(8e5)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a tuple of a batch's input and output data."""
        # Rotate index if necessary
        index = index % self.n_samples

        # Get the input image. Try several indices if necessary.
        prefix_attempt = 0
        prefix_loop_break = False
        while prefix_attempt < 5:
            file_prefix = self.file_prefix_list[index]

            # Use `while` loop to search until a pair of files which exists is
            # found.
            attempt = 0
            while attempt < self.max_file_checks:
                # Files have name format
                # *lsc240420_id01001_pvi_idx00000.npz*.
                #
                # Choose random starting index 0-(100-max_time_idx_offset) so
                # the end index will be less than or equal to 99.
                start_idx = self.rng.integers(0, 100 - self.max_time_idx_offset)
                end_idx = self.rng.integers(0, self.max_time_idx_offset + 1) + start_idx

                # Construct file names
                start_file = file_prefix + f"_pvi_idx{start_idx:05d}.npz"
                end_file = file_prefix + f"_pvi_idx{end_idx:05d}.npz"

                # Check if both files exist
                start_file_path = Path(self.npz_dir + start_file)
                end_file_path = Path(self.npz_dir + end_file)

                if start_file_path.is_file() and end_file_path.is_file():
                    prefix_loop_break = True
                    break

                attempt += 1

            if attempt == self.max_file_checks:
                fnf_msg = (
                    "In TemporalDataSet, "
                    "max_file_checks "
                    f"reached for prefix: {file_prefix}"
                )
                print(fnf_msg, file=sys.stderr)

            # Break outer loop if time-pairs were found.
            if prefix_loop_break:
                break

            # Try different prefix if no time-pairs are found.
            print(
                f"Prefix attempt {prefix_attempt + 1} failed. Trying next prefix.",
                file=sys.stderr,
            )
            prefix_attempt += 1
            index = (index + 1) % self.n_samples  # Rotate index if necessary

        # Load NPZ files. Raise exceptions if file is not able to be loaded.
        try:
            start_npz = np.load(self.npz_dir + start_file)

            # These will change from simulation key to key:
            self.active_npz_field_names = LabeledData(
                self.npz_dir + start_file, self.csv_filepath
            ).get_active_npz_field_names()
            active_hydro_field_names = LabeledData(
                self.npz_dir + start_file, self.csv_filepath
            ).get_active_hydro_field_names()
            channel_map = LabeledData(
                self.npz_dir + start_file, self.csv_filepath
            ).get_channel_map()

        except Exception as e:
            print(
                f"Error loading start file: {self.npz_dir + start_file}",
                file=sys.stderr,
            )
            raise e

        try:
            end_npz = np.load(self.npz_dir + end_file)

            # For now, we assume that the start and end files have the same materials,
            # etc. (it's hard to imagine a physical scenario in which Cu emerges from
            # nothing, but hey, we're not comparing with experiment so anything goes!)

        except Exception as e:
            print(
                f"Error loading end file: {self.npz_dir + end_file}",
                file=sys.stderr,
            )
            start_npz.close()
            raise e

        start_img_list = []
        end_img_list = []

        for hfield in self.active_npz_field_names:
            # start npz
            tmp_img = import_img_from_npz(start_npz, hfield)
            if not self.half_image:
                tmp_img = np.concatenate((np.fliplr(tmp_img), tmp_img), axis=1)
            start_img_list.append(tmp_img)

            # end_npz
            tmp_img = import_img_from_npz(end_npz, hfield)
            if not self.half_image:
                tmp_img = np.concatenate((np.fliplr(tmp_img), tmp_img), axis=1)
            end_img_list.append(tmp_img)

        img_list_combined = np.array([start_img_list, end_img_list])
        channel_map, img_list_combined, active_hydro_field_names = process_channel_data(
            channel_map, img_list_combined, active_hydro_field_names
        )
        start_img_list = img_list_combined[0]
        end_img_list = img_list_combined[1]
        self.channel_map = channel_map
        self.active_hydro_field_names = active_hydro_field_names

        # Concatenate images channel first.
        start_img = torch.tensor(np.stack(start_img_list, axis=0)).to(torch.float32)
        end_img = torch.tensor(np.stack(end_img_list, axis=0)).to(torch.float32)

        # Get the time offset
        dt = torch.tensor(0.25 * (end_idx - start_idx), dtype=torch.float32)

        # Close the npzs
        start_npz.close()
        end_npz.close()

        return start_img, self.channel_map, end_img, self.channel_map, dt


class SequentialDataSet(Dataset):
    """Returns a sequence of consecutive frames from a simulation.

    For example, if seq_len=4, you'll get frames t, t+1, t+2, t+3.

    Args:
        npz_dir (str): Directory storing NPZ files of the dataset being analyzed.
        csv_filepath (str): Path to the 'design' file (CSV).
        file_prefix_list (str): Text file listing unique prefixes corresponding
                                to unique simulations.
        max_file_checks (int): Maximum number of attempts to find valid file sequences.
        seq_len (int): Number of consecutive frames to return. This includes the
                       starting frame.
        half_image (bool): If True, returns half-images, otherwise full images.

    """

    def __init__(
        self,
        npz_dir: str,
        csv_filepath: str,
        file_prefix_list: str,
        max_file_checks: int,
        seq_len: int,
        half_image: bool = True,
    ) -> None:
        """Initialization for sequential dataset."""
        dir_path = Path(npz_dir)
        # Ensure the directory exists and is indeed a directory
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {npz_dir}")

        self.npz_dir = npz_dir
        self.csv_filepath = csv_filepath
        self.max_file_checks = max_file_checks
        self.seq_len = seq_len
        self.half_image = half_image

        # Load the list of file prefixes
        with open(file_prefix_list, encoding="utf-8") as f:
            self.file_prefix_list = [line.rstrip() for line in f]

        # Shuffle the prefixes for randomness
        random.shuffle(self.file_prefix_list)
        self.n_samples = len(self.file_prefix_list)

        # Random number generator
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a sequence of consecutive frames."""
        # Rotate index if necessary
        index = index % self.n_samples
        file_prefix = self.file_prefix_list[index]

        # Try multiple attempts to find valid files
        prefix_attempt = 0
        while prefix_attempt < self.max_file_checks:
            # Pick a random start index so that the sequence fits within the range
            start_idx = self.rng.integers(0, 100 - self.seq_len)

            # Construct the sequence of file paths
            valid_sequence = True
            file_paths = []
            for offset in range(self.seq_len):
                idx = start_idx + offset
                file_name = f"{file_prefix}_pvi_idx{idx:05d}.npz"
                file_path = Path(self.npz_dir, file_name)

                if not file_path.is_file():
                    valid_sequence = False
                    break

                file_paths.append(file_path)

            if valid_sequence:
                break

            # If no valid sequence found, try the next prefix
            prefix_attempt += 1
            index = (index + 1) % self.n_samples  # Rotate index to try another prefix

        if prefix_attempt == self.max_file_checks:
            err_msg = (
                f"Failed to find valid sequence for prefix: {file_prefix} "
                f"after {self.max_file_checks} attempts."
            )
            raise RuntimeError(err_msg)

        # Load and process the sequence of frames
        frames = []
        for file_path in file_paths:
            try:
                data_npz = np.load(file_path)

                # Fields to extract from the simulation
                self.active_npz_field_names = LabeledData(
                    file_path, self.csv_filepath
                ).get_active_npz_field_names()
                active_hydro_field_names = LabeledData(
                    file_path, self.csv_filepath
                ).get_active_hydro_field_names()
                channel_map = LabeledData(file_path, self.csv_filepath).get_channel_map()

            except Exception as e:
                raise RuntimeError(f"Error loading file: {file_path}") from e

            field_imgs = []
            for hfield in self.active_npz_field_names:
                tmp_img = import_img_from_npz(data_npz, hfield)
                if not self.half_image:
                    tmp_img = np.concatenate((np.fliplr(tmp_img), tmp_img), axis=1)
                field_imgs.append(tmp_img)

            data_npz.close()

            img_list_combined = np.array([field_imgs])
            channel_map, img_list_combined, active_hydro_field_names = (
                process_channel_data(
                    channel_map, img_list_combined, active_hydro_field_names
                )
            )
            field_imgs = img_list_combined[0]

            # Stack the fields for this frame
            field_tensor = torch.tensor(
                np.stack(field_imgs, axis=0), dtype=torch.float32
            )
            frames.append(field_tensor)

        self.channel_map = channel_map
        self.active_hydro_field_names = active_hydro_field_names

        # Combine frames into a single tensor of shape [seq_len, num_fields, H, W]
        img_seq = torch.stack(frames, dim=0)

        # Fixed time offset
        dt = torch.tensor(0.25, dtype=torch.float32)

        return img_seq, dt, self.channel_map
