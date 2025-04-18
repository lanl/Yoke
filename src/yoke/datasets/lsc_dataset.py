"""Relating to the layered shaped charge data.

Functions and classes for torch DataSets which sample the Layered Shaped Charge
data, *lsc240420*.

"""

####################################
# Packages
####################################
import itertools
from pathlib import Path
import random
import sys
import typing
from typing import Callable

import lightning.pytorch as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


NoneStr = typing.Union[None, str]


################################################
# Functions for returning the run *key* from
# the npz-file name
################################################
def LSCnpz2key(npz_file: str) -> str:
    """Simulation key extraction.

    Function to extract simulation *key* from the name of an .npz file.

    A study key looks like **lsc240420_id00001** and a NPZ filename is like
    **lsc240420_id00001_pvi_idx00000.npz**

    Args:
        npz_file (str): file path from working directory to .npz file

    Returns:
        key (str): The correspond simulation key for the NPZ file.

    """
    key = npz_file.split("/")[-1].split("_pvi_")[0]

    return key


def LSCcsv2bspline_pts(design_file: str, key: str) -> np.ndarray:
    """Function to extract the B-spline nodes.

    Nodes are extracted from the design .csv file given the study key.

    Args:
        design_file (str): File path from working directory to the .csv design file
        key (str): The study information for a given simulation; of the
                   form *lsc240420_id?????*

    Returns:
        bspline_pts (numpy array): The B-spline nodes defining the geometry of
                                   the Layered Shaped Charge

    """
    design_df = pd.read_csv(design_file, sep=",", header=0, index_col=0, engine="python")

    # removed spaces from headers
    for col in design_df.columns:
        design_df.rename(columns={col: col.strip()}, inplace=True)

    bspline_pts = design_df.loc[key, "sa1":"ct7"].values

    return bspline_pts.astype(float)


def LSCread_npz(npz: np.lib.npyio.NpzFile, field: str) -> np.ndarray:
    """Function to extract a value corresponding to an NPZ key.

    Args:
        npz (np.lib.npyio.NpzFile): a loaded .npz file
        field (str): name of field to extract

    """
    return npz[field]


def LSCread_npz_NaN(npz: np.lib.npyio.NpzFile, field: str) -> np.ndarray:
    """Extract a specific field from a .npz file and replace NaNs with 0.

    Args:
        npz (np.lib.npyio.NpzFile): Loaded .npz file.
        field (str): Field name to extract.

    Returns:
        np.ndarray: Field data with NaNs replaced by 0.

    """
    return np.nan_to_num(npz[field], nan=0.0)


class LSC_cntr2rho_DataSet(Dataset):
    """Contour to average density dataset."""

    def __init__(self, LSC_NPZ_DIR: str, filelist: str, design_file: str) -> None:
        """Initialization of class.

        The definition of a dataset object for the *Layered Shaped Charge* data
        which produces pairs of B-spline contour-node vectors and simulation
        times together with an average density field.

        Args:
            LSC_NPZ_DIR (str): Location of LSC NPZ files. A YOKE env variable.
            filelist (str): Text file listing file names to read
            design_file (str): .csv file with master design study parameters

        """
        # Model Arguments
        self.LSC_NPZ_DIR = LSC_NPZ_DIR
        self.filelist = filelist
        self.design_file = design_file

        # Create filelist
        with open(filelist) as f:
            self.filelist = [line.rstrip() for line in f]

        self.Nsamples = len(self.filelist)

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return self.Nsamples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a tuple of a batch's input and output data."""
        # Rotate index if necessary
        index = index % self.Nsamples

        # Get the input image
        filepath = self.filelist[index]
        npz = np.load(self.LSC_NPZ_DIR + filepath)

        true_image = LSCread_npz(npz, "av_density")
        true_image = np.concatenate((np.fliplr(true_image), true_image), axis=1)
        nY, nX = true_image.shape
        true_image = true_image.reshape((1, nY, nX))
        true_image = torch.tensor(true_image).to(torch.float32)

        # Get the contours and sim_time
        sim_key = LSCnpz2key(self.LSC_NPZ_DIR + filepath)
        Bspline_nodes = LSCcsv2bspline_pts(self.design_file, sim_key)
        sim_time = npz["sim_time"]
        npz.close()

        sim_params = np.append(Bspline_nodes, sim_time)
        sim_params = torch.from_numpy(sim_params).to(torch.float32)

        return sim_params, true_image


class LSCnorm_cntr2rho_DataSet(Dataset):
    """Normalized contour to average density dataset."""

    def __init__(
        self, LSC_NPZ_DIR: str, filelist: str, design_file: str, normalization_file: str
    ) -> None:
        """Initialization of normalized dataset.

        The definition of a dataset object for the *Layered Shaped Charge* data
        which produces pairs of B-spline contour-node vectors and simulation
        times together with an average density field.

        This class uses pre-calculated normalization constants to normalize the
        geometry parameters and the density field. The time values are also
        normalized to [0, 1] at regular intervals of 0.25 us.

        This class relies on a normalization file created by the script
        `applications/normalization/generate_lsc_normalization.py`

        The normalization file is an NPZ created by the call:

        np.savez('./lsc240420_norm.npz',
                 image_avg=image_avg,
                 image_min=image_min,
                 image_max=image_max,
                 Bspline_avg=Bspline_avg,
                 Bspline_min=Bspline_min,
                 Bspline_max=Bspline_max,
                 **avg_time_dict)

        Args:
            LSC_NPZ_DIR (str): Location of LSC NPZ files. A YOKE env variable.
            filelist (str): Text file listing file names to read
            design_file (str): .csv file with master design study parameters
            normalization_file: Full-path to the NPZ file containing the pre-calculated
                                normalization quantities.

        """
        # Model Arguments
        self.LSC_NPZ_DIR = LSC_NPZ_DIR
        self.filelist = filelist
        self.design_file = design_file
        self.normalization_file = normalization_file

        norm_npz = np.load(self.normalization_file)
        time_keys = [k for k in norm_npz.keys()]
        time_keys.remove("image_avg")
        time_keys.remove("image_min")
        time_keys.remove("image_max")
        time_keys.remove("Bspline_avg")
        time_keys.remove("Bspline_min")
        time_keys.remove("Bspline_max")
        self.time_keys = sorted([float(k) for k in time_keys])

        self.avg_rho_by_time = dict()
        for tk in self.time_keys:
            self.avg_rho_by_time[str(tk)] = norm_npz[str(tk)]

        self.Bspline_min = norm_npz["Bspline_min"]
        self.Bspline_max = norm_npz["Bspline_max"]

        # Create filelist
        with open(filelist) as f:
            self.filelist = [line.rstrip() for line in f]

        self.Nsamples = len(self.filelist)

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return self.Nsamples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a tuple of a batch's input and output data."""
        # Rotate index if necessary
        index = index % self.Nsamples

        # Get the input image
        filepath = self.filelist[index]
        npz = np.load(self.LSC_NPZ_DIR + filepath)

        # Get time associated with image
        sim_time = npz["sim_time"]
        round_sim_time = round(4.0 * sim_time) / 4.0

        # Load image
        true_image = LSCread_npz(npz, "av_density")
        true_image = np.concatenate((np.fliplr(true_image), true_image), axis=1)
        # unbias image
        unbias_true_image = true_image - self.avg_rho_by_time[str(round_sim_time)]
        # unbias_true_image = self.avg_rho_by_time[str(round_sim_time)]
        nY, nX = unbias_true_image.shape
        unbias_true_image = unbias_true_image.reshape((1, nY, nX))
        unbias_true_image = torch.tensor(unbias_true_image).to(torch.float32)

        # Get the contours and sim_time
        sim_key = LSCnpz2key(self.LSC_NPZ_DIR + filepath)
        Bspline_nodes = LSCcsv2bspline_pts(self.design_file, sim_key)

        npz.close()

        # Scale round_sim_time
        norm_time = round_sim_time / 25.0

        # Normalize Bspline nodes
        norm_Bspline_nodes = (Bspline_nodes - self.Bspline_min) / (
            self.Bspline_max - self.Bspline_min
        )
        norm_sim_params = np.append(norm_Bspline_nodes, norm_time)
        norm_sim_params = torch.from_numpy(norm_sim_params).to(torch.float32)

        return norm_sim_params, unbias_true_image


class LSC_cntr2hfield_DataSet(Dataset):
    """Contour to set of fields dataset."""

    def __init__(
        self,
        LSC_NPZ_DIR: str,
        filelist: str,
        design_file: str,
        field_list: list[str] = ["density_throw"],
    ) -> None:
        """Initialization of class.

        The definition of a dataset object for the *Layered Shaped Charge* data
        which produces B-spline contour-node vectors and a *hydro-dynamic
        field* image consisting of channels specified in *field_list*.

        Args:
            LSC_NPZ_DIR (str): Location of LSC NPZ files. A YOKE env variable.
            filelist (str): Text file listing file names to read
            design_file (str): Full-path to .csv file with master design study parameters
            field_list (List[str]): List of hydro-dynamic fields to include as channels
                                    in image.

        """
        # Model Arguments
        self.LSC_NPZ_DIR = LSC_NPZ_DIR
        self.filelist = filelist
        self.design_file = design_file
        self.hydro_fields = field_list

        # Create filelist
        with open(filelist) as f:
            self.filelist = [line.rstrip() for line in f]

        # Shuffle the list of prefixes in-place
        random.shuffle(self.filelist)

        self.Nsamples = len(self.filelist)

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return self.Nsamples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a tuple of a batch's input and output data."""
        # Rotate index if necessary
        index = index % self.Nsamples

        # Get the input image
        filepath = self.filelist[index]
        npz = np.load(self.LSC_NPZ_DIR + filepath)

        hfield_list = []
        for hfield in self.hydro_fields:
            tmp_img = LSCread_npz_NaN(npz, hfield)
            hfield_list.append(tmp_img)

        # Concatenate images channel first.
        hfield = torch.tensor(np.stack(hfield_list, axis=0)).to(torch.float32)

        # Get the contours and sim_time
        sim_key = LSCnpz2key(self.LSC_NPZ_DIR + filepath)
        Bspline_nodes = LSCcsv2bspline_pts(self.design_file, sim_key)
        npz.close()

        geom_params = torch.from_numpy(Bspline_nodes).to(torch.float32)

        return geom_params, hfield


def neg_mse_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Negated MSE loss."""
    return -torch.nn.functional.mse_loss(x, y)


class LSC_hfield_reward_DataSet(Dataset):
    """Hydro-field discrepancy reward dataset.

    The definition of a dataset object for the *Layered Shaped Charge* data
    which produces tuples `(y', H', H*, -MSE(H', H*))`. `y'` is the vector
    of B-spline contour-nodes. `H'` is the tensor of hydro-fields at final
    time corresponding to `y'`. `H*` is a *target* tensor of hydro-fields
    at final time, chosen randomly from the available training data.

    A *value* network will be pre-trained from this dataset to use in a
    *proximal policy optimization* (PPO) reinforcement learning algorithm.

    Args:
        LSC_NPZ_DIR (str): Location of LSC NPZ files. A YOKE env variable.
        filelist (str): Text file listing file names to read
        design_file (str): Full-path to .csv file with master design study parameters
        half_image (bool): If True then returned images are NOT reflected about axis
                           of symmetry and half-images are returned instead.
        field_list (list[str]): List of hydro-dynamic fields to include as channels
                                in image.
        reward_fn (Callable): Function taking two torch.tensor and returning a
                              scalar reward.

    """

    def __init__(
        self,
        LSC_NPZ_DIR: str,
        filelist: str,
        design_file: str,
        half_image: bool = True,
        field_list: list[str] = ["density_throw"],
        reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = neg_mse_loss,
    ) -> None:
        """Initialization of class."""
        # Model Arguments
        self.LSC_NPZ_DIR = LSC_NPZ_DIR
        self.filelist = filelist
        self.design_file = design_file
        self.half_image = half_image
        self.hydro_fields = field_list
        self.reward = reward_fn

        # Create filelist
        with open(filelist) as f:
            self.filelist = [line.rstrip() for line in f]

        # Create list of state-target pairs.
        self.state_target_list = list(itertools.product(self.filelist, self.filelist))

        # Shuffle the list of state-target pairs in-place
        random.shuffle(self.state_target_list)

        self.Nsamples = len(self.state_target_list)

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return int(1e6)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a tuple of a batch's input and output data."""
        # Rotate index if necessary
        index = index % self.Nsamples

        # Get the state-target pair
        state, target = self.state_target_list[index]
        state_npz = np.load(self.LSC_NPZ_DIR + state)
        target_npz = np.load(self.LSC_NPZ_DIR + target)

        state_hfield_list = []
        target_hfield_list = []
        for hfield in self.hydro_fields:
            tmp_img = LSCread_npz_NaN(state_npz, hfield)
            if not self.half_image:
                tmp_img = np.concatenate((np.fliplr(tmp_img), tmp_img), axis=1)
            state_hfield_list.append(tmp_img)

            tmp_img = LSCread_npz_NaN(target_npz, hfield)
            if not self.half_image:
                tmp_img = np.concatenate((np.fliplr(tmp_img), tmp_img), axis=1)
            target_hfield_list.append(tmp_img)

        # Concatenate images channel first.
        state_hfield = torch.tensor(np.stack(state_hfield_list, axis=0)).to(
            torch.float32
        )

        target_hfield = torch.tensor(np.stack(target_hfield_list, axis=0)).to(
            torch.float32
        )

        # Calculate reward.
        #
        # Make sure the reward computation isn't part of the torch
        # computational graph which could happen if the reward function is a
        # torch Loss.
        with torch.no_grad():
            reward = self.reward(state_hfield, target_hfield)

        # Get the state contours
        sim_key = LSCnpz2key(self.LSC_NPZ_DIR + state)
        Bspline_nodes = LSCcsv2bspline_pts(self.design_file, sim_key)
        state_npz.close()
        target_npz.close()

        state_geom_params = torch.from_numpy(Bspline_nodes).to(torch.float32)

        return state_geom_params, state_hfield, target_hfield, reward


class LSC_hfield_policy_DataSet(Dataset):
    """Hydro-field policy dataset.

    The definition of a dataset object for the *Layered Shaped Charge* data
    which produces tuples `(y', H', H*, x=y*-y')`. `y'` is the vector of
    B-spline contour-nodes. `H'` is the tensor of hydro-fields at final
    time corresponding to `y'`. `H*` is a *target* tensor of hydro-fields
    at final time, chosen randomly from the available training data. The
    optimal *policy*, `x=y* - y'` is the prediction goal for this dataset.

    A *policy* network will be pre-trained from this dataset to use in a
    *proximal policy optimization* (PPO) reinforcement learning algorithm.

    Args:
        LSC_NPZ_DIR (str): Location of LSC NPZ files. A YOKE env variable.
        filelist (str): Text file listing file names to read
        design_file (str): Full-path to .csv file with master design study parameters
        half_image (bool): If True then returned images are NOT reflected about axis
                           of symmetry and half-images are returned instead.
        field_list (list[str]): List of hydro-dynamic fields to include as channels
                                in image.

    """

    def __init__(
        self,
        LSC_NPZ_DIR: str,
        filelist: str,
        design_file: str,
        half_image: bool = True,
        field_list: list[str] = ["density_throw"],
    ) -> None:
        """Initialization of class."""
        # Model Arguments
        self.LSC_NPZ_DIR = LSC_NPZ_DIR
        self.filelist = filelist
        self.design_file = design_file
        self.half_image = half_image
        self.hydro_fields = field_list

        # Create filelist
        with open(filelist) as f:
            self.filelist = [line.rstrip() for line in f]

        # Create list of state-target pairs.
        self.state_target_list = list(itertools.product(self.filelist, self.filelist))

        # Shuffle the list of state-target pairs in-place
        random.shuffle(self.state_target_list)

        self.Nsamples = len(self.state_target_list)

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return int(1e6)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a tuple of a batch's input and output data."""
        # Rotate index if necessary
        index = index % self.Nsamples

        # Get the state-target pair
        state, target = self.state_target_list[index]
        state_npz = np.load(self.LSC_NPZ_DIR + state)
        target_npz = np.load(self.LSC_NPZ_DIR + target)

        state_hfield_list = []
        target_hfield_list = []
        for hfield in self.hydro_fields:
            tmp_img = LSCread_npz_NaN(state_npz, hfield)
            if not self.half_image:
                tmp_img = np.concatenate((np.fliplr(tmp_img), tmp_img), axis=1)
            state_hfield_list.append(tmp_img)

            tmp_img = LSCread_npz_NaN(target_npz, hfield)
            if not self.half_image:
                tmp_img = np.concatenate((np.fliplr(tmp_img), tmp_img), axis=1)

            target_hfield_list.append(tmp_img)

        # Concatenate images channel first.
        state_hfield = torch.tensor(np.stack(state_hfield_list, axis=0)).to(
            torch.float32
        )

        target_hfield = torch.tensor(np.stack(target_hfield_list, axis=0)).to(
            torch.float32
        )

        # Get the contour parameters
        sim_key = LSCnpz2key(self.LSC_NPZ_DIR + state)
        Bspline_nodes = LSCcsv2bspline_pts(self.design_file, sim_key)
        state_geom_params = torch.from_numpy(Bspline_nodes).to(torch.float32)

        sim_key = LSCnpz2key(self.LSC_NPZ_DIR + target)
        Bspline_nodes = LSCcsv2bspline_pts(self.design_file, sim_key)
        target_geom_params = torch.from_numpy(Bspline_nodes).to(torch.float32)

        state_npz.close()
        target_npz.close()

        # Calculate optimal policy step
        geom_discrepancy = target_geom_params - state_geom_params

        return state_geom_params, state_hfield, target_hfield, geom_discrepancy


class LSC_rho2rho_temporal_DataSet(Dataset):
    """Temporal LSC dataset."""

    def __init__(
        self,
        LSC_NPZ_DIR: str,
        file_prefix_list: str,
        max_timeIDX_offset: int,
        max_file_checks: int,
        half_image: bool = True,
        hydro_fields: np.array = np.array(
            [
                "density_case",
                "density_cushion",
                "density_maincharge",
                "density_outside_air",
                "density_striker",
                "density_throw",
                "Uvelocity",
                "Wvelocity",
            ]
        ),
    ) -> None:
        """Initialization of timestep dataset.

        This dataset returns multi-channel images at two different times from
        the *Layered Shaped Charge* simulation. The *maximum time-offset* can
        be specified. The channels in the images returned are the densities for
        each material at a given time as well as the (R, Z)-velocity
        fields. The time-offset between the two images is also returned.

        NOTE: The way time indices are chosen necessitates *max_timeIDX_offset*
        being less than or equal to 3 in the lsc240420 data.

        Args:
            LSC_NPZ_DIR (str): Location of LSC NPZ files.
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
            hydro_fields (np.array, optional): Array of hydro field names to be included.
                                               Defaults to:
                                               [
                                                   "density_case",
                                                   "density_cushion",
                                                   "density_maincharge",
                                                   "density_outside_air",
                                                   "density_striker",
                                                   "density_throw",
                                                   "Uvelocity",
                                                   "Wvelocity",
                                               ].

        """
        # Model Arguments
        self.LSC_NPZ_DIR = LSC_NPZ_DIR
        self.max_timeIDX_offset = max_timeIDX_offset
        self.max_file_checks = max_file_checks
        self.half_image = half_image

        # Create filelist
        with open(file_prefix_list) as f:
            self.file_prefix_list = [line.rstrip() for line in f]

        # Shuffle the list of prefixes in-place
        random.shuffle(self.file_prefix_list)

        self.Nsamples = len(self.file_prefix_list)

        self.hydro_fields = hydro_fields

        # Initialize random number generator for time index selection
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        """Return effectively infinite number of samples in dataset."""
        return int(1e6)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a tuple of a batch's input and output data."""
        # Rotate index if necessary
        index = index % self.Nsamples

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
                # Choose random starting index 0-(100-max_timeIDX_offset) so
                # the end index will be less than or equal to 99.
                startIDX = self.rng.integers(0, 100 - self.max_timeIDX_offset)
                endIDX = self.rng.integers(0, self.max_timeIDX_offset + 1) + startIDX

                # Construct file names
                start_file = file_prefix + f"_pvi_idx{startIDX:05d}.npz"
                end_file = file_prefix + f"_pvi_idx{endIDX:05d}.npz"

                # Check if both files exist
                start_file_path = Path(self.LSC_NPZ_DIR + start_file)
                end_file_path = Path(self.LSC_NPZ_DIR + end_file)

                if start_file_path.is_file() and end_file_path.is_file():
                    prefix_loop_break = True
                    break

                attempt += 1

            if attempt == self.max_file_checks:
                fnf_msg = (
                    "In LSC_rho2rho_temporal_DataSet, "
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
            index = (index + 1) % self.Nsamples  # Rotate index if necessary

        # Load NPZ files. Raise exceptions if file is not able to be loaded.
        try:
            start_npz = np.load(self.LSC_NPZ_DIR + start_file)
        except Exception as e:
            print(
                f"Error loading start file: {self.LSC_NPZ_DIR + start_file}",
                file=sys.stderr,
            )
            raise e

        try:
            end_npz = np.load(self.LSC_NPZ_DIR + end_file)
        except Exception as e:
            print(
                f"Error loading end file: {self.LSC_NPZ_DIR + end_file}",
                file=sys.stderr,
            )
            start_npz.close()
            raise e

        start_img_list = []
        end_img_list = []
        for hfield in self.hydro_fields:
            tmp_img = LSCread_npz_NaN(start_npz, hfield)
            if not self.half_image:
                tmp_img = np.concatenate((np.fliplr(tmp_img), tmp_img), axis=1)
            start_img_list.append(tmp_img)

            tmp_img = LSCread_npz_NaN(end_npz, hfield)
            if not self.half_image:
                tmp_img = np.concatenate((np.fliplr(tmp_img), tmp_img), axis=1)
            end_img_list.append(tmp_img)

        # Concatenate images channel first.
        start_img = torch.tensor(np.stack(start_img_list, axis=0)).to(torch.float32)
        end_img = torch.tensor(np.stack(end_img_list, axis=0)).to(torch.float32)

        # Get the time offset
        Dt = torch.tensor(0.25 * (endIDX - startIDX), dtype=torch.float32)

        # Close the npzs
        start_npz.close()
        end_npz.close()

        return start_img, end_img, Dt


class LSC_rho2rho_sequential_DataSet(Dataset):
    """Returns a sequence of consecutive frames from the LSC simulation.

    For example, if seq_len=4, you'll get frames t, t+1, t+2, t+3.

    Args:
        LSC_NPZ_DIR (str): Location of LSC NPZ files.
        file_prefix_list (str): Text file listing unique prefixes corresponding
                                to unique simulations.
        max_file_checks (int): Maximum number of attempts to find valid file sequences.
        seq_len (int): Number of consecutive frames to return. This includes the
                       starting frame.
        half_image (bool): If True, returns half-images, otherwise full images.
        hydro_fields (np.array, optional): Array of hydro field names to be included.
        transform (Callable): Transform applied to loaded data sequence before returning.
    """

    def __init__(
        self,
        LSC_NPZ_DIR: str,
        file_prefix_list: str,
        max_file_checks: int,
        seq_len: int,
        half_image: bool = True,
        hydro_fields: np.array = np.array(
            [
                "density_case",
                "density_cushion",
                "density_maincharge",
                "density_outside_air",
                "density_striker",
                "density_throw",
                "Uvelocity",
                "Wvelocity",
            ]
        ),
        transform: Callable = None,
    ) -> None:
        """Initialization for LSC sequential dataset."""
        dir_path = Path(LSC_NPZ_DIR)
        # Ensure the directory exists and is indeed a directory
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {LSC_NPZ_DIR}")

        self.LSC_NPZ_DIR = LSC_NPZ_DIR
        self.max_file_checks = max_file_checks
        self.seq_len = seq_len
        self.half_image = half_image
        self.transform = transform

        # Load the list of file prefixes
        with open(file_prefix_list) as f:
            self.file_prefix_list = [line.rstrip() for line in f]

        # Shuffle the prefixes for randomness
        random.shuffle(self.file_prefix_list)
        self.Nsamples = len(self.file_prefix_list)

        # Fields to extract from the simulation
        self.hydro_fields = hydro_fields

        # Random number generator
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.Nsamples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a sequence of consecutive frames."""
        # Rotate index if necessary
        index = index % self.Nsamples
        file_prefix = self.file_prefix_list[index]

        # Try multiple attempts to find valid files
        prefix_attempt = 0
        while prefix_attempt < self.max_file_checks:
            # Pick a random start index so that the sequence fits within the range
            startIDX = self.rng.integers(0, 100 - self.seq_len)

            # Construct the sequence of file paths
            valid_sequence = True
            file_paths = []
            for offset in range(self.seq_len):
                idx = startIDX + offset
                file_name = f"{file_prefix}_pvi_idx{idx:05d}.npz"
                file_path = Path(self.LSC_NPZ_DIR, file_name)

                if not file_path.is_file():
                    valid_sequence = False
                    break

                file_paths.append(file_path)

            if valid_sequence:
                break

            # If no valid sequence found, try the next prefix
            prefix_attempt += 1
            index = (index + 1) % self.Nsamples  # Rotate index to try another prefix

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
            except Exception as e:
                raise RuntimeError(f"Error loading file: {file_path}") from e

            field_imgs = []
            for hfield in self.hydro_fields:
                tmp_img = LSCread_npz_NaN(data_npz, hfield)

                # Reflect image if not half_image
                if not self.half_image:
                    tmp_img = np.concatenate((np.fliplr(tmp_img), tmp_img), axis=1)

                field_imgs.append(tmp_img)

            data_npz.close()

            # Stack the fields for this frame
            field_tensor = torch.tensor(
                np.stack(field_imgs, axis=0), dtype=torch.float32
            )
            frames.append(field_tensor)

        # Combine frames into a single tensor of shape [seq_len, num_fields, H, W]
        img_seq = torch.stack(frames, dim=0)

        # Apply transforms if requested.
        if self.transform is not None:
            img_seq = self.transform(img_seq)

        # Fixed time offset
        Dt = torch.tensor(0.25, dtype=torch.float32)

        return img_seq, Dt


class LSCDataModule(L.LightningDataModule):
    """Lightning data module for generic LSC datasets.

    Args:
        ds_name (str): Name of desired dataset in src.yoke.datasets.lsc_dataset
        ds_params_train (dict): Keyword arguments passed to dataset initializer
            to generate the training dataset.
        dl_params_train (dict): Keyword arguments passed to training dataloader.
        ds_params_val (dict): Keyword arguments passed to dataset initializer
            to generate the validation dataset.
        dl_params_val (dict): Keyword arguments passed to validation dataloader.
        ds_params_test (dict): Keyword arguments passed to dataset initializer
            to generate the testing dataset.
        dl_params_test (dict): Keyword arguments passed to testing dataloader.
    """

    def __init__(
        self,
        ds_name: str,
        ds_params_train: dict,
        dl_params_train: dict,
        ds_params_val: dict,
        dl_params_val: dict,
        ds_params_test: dict = None,
        dl_params_test: dict = None,
    ) -> None:
        """LSCDataModule initialization method."""
        super().__init__()
        self.ds_name = ds_name

        self.ds_params_train = ds_params_train
        self.dl_params_train = dl_params_train

        self.ds_params_val = ds_params_val
        self.dl_params_val = dl_params_val

        self.ds_params_test = ds_params_test
        self.dl_params_test = dl_params_test

    def setup(self, stage: str = None) -> None:
        """Data module setup called on all devices."""
        # Currently, LSC datasets are fast to instantiate.  As such, to
        # facilitate "dynamic" datasets that may change throughout
        # training, dataset instantiation is done on-the-fly when
        # preparing dataloaders.
        pass

    def train_dataloader(self) -> DataLoader:
        """Prepare the training dataset and dataloader."""
        ds_ref = getattr(sys.modules[__name__], self.ds_name)
        ds_train = ds_ref(**self.ds_params_train)
        self.ds_train = ds_train
        return DataLoader(dataset=ds_train, **self.dl_params_train)

    def val_dataloader(self) -> DataLoader:
        """Prepare the validation dataset and dataloader."""
        ds_ref = getattr(sys.modules[__name__], self.ds_name)
        ds_val = ds_ref(**self.ds_params_val)
        self.ds_val = ds_val
        return DataLoader(dataset=ds_val, **self.dl_params_val)

    def test_dataloader(self) -> DataLoader:
        """Prepare the testing dataset and dataloader."""
        ds_ref = getattr(sys.modules[__name__], self.ds_name)
        ds_test = ds_ref(**self.ds_params_test)
        self.ds_test = ds_test
        return DataLoader(dataset=ds_test, **self.dl_params_test)
