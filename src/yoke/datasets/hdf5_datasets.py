"""https://github.com/PolymathicAI/multiple_physics_pretraining."""

import torch
import torch.nn
import numpy as np
from torch.utils.data import Dataset
import h5py
from typing import Optional
import glob

broken_paths = [""]


class BaseHDF5DirectoryDataset(Dataset):
    """Base class for data loaders. Returns data in T x B x C x H x W format.

    Note - doesn't currently normalize because the data is on wildly different
    scales but probably should.

    Split is provided so I can be lazy and not separate out HDF5 files.

    Takes in path to directory of HDF5 files to construct dset.

    Args:
        path (str): Path to directory of HDF5 files
        include_string (str): Only include files with this string in name
        n_steps (int): Number of steps to include in each sample
        dt (int): Time step between samples
        split (str): train/val/test split
        train_val_test (tuple): Percent of data to use for train/val/test
        subname (str): Name to use for dataset
        split_level (str): 'sample' or 'file' - whether to split by samples
            within a file (useful for data segmented by parameters) or file
            (mostly INS right now)
    """

    def __init__(
        self,
        path: str,
        include_string: str = "",
        n_steps: int = 1,
        dt: int = 1,
        split: str = "train",
        train_val_test: Optional[tuple[float, float, float]] = None,
        subname: Optional[str] = None,
        extra_specific: bool = False,
    ) -> None:
        """No docstring."""
        super().__init__()
        self.path = path
        self.split = split
        self.extra_specific = extra_specific
        self.subname = path.split("/")[-1] if subname is None else subname
        self.dt = 1
        self.n_steps = n_steps
        self.include_string = include_string
        self.train_val_test = train_val_test
        self.partition = {"train": 0, "val": 1, "test": 2}[split]
        (
            self.time_index,
            self.sample_index,
            self.field_names,
            self.type,
            self.split_level,
        ) = self._specifics()
        self._get_directory_stats(path)
        self.title = (
            self.more_specific_title(self.type, path, include_string)
            if self.extra_specific
            else self.type
        )

    def get_name(self, full_name: bool = False) -> str:
        """No docstring."""
        return f"{self.subname}_{self.type}" if full_name else self.type

    def more_specific_title(self, type: str, path: str, include_string: str) -> str:
        """Override this to add more info to the dataset name."""
        return type

    @staticmethod
    def _specifics() -> tuple:
        """Sets self.field_names, self.dataset_type."""
        raise NotImplementedError

    def get_per_file_dsets(self) -> list:
        """No docstring."""
        if self.split_level == "file" or len(self.files_paths) == 1:
            return [self]
        sub_dsets = []
        for file in self.files_paths:
            subd = self.__class__(
                self.path,
                file,
                n_steps=self.n_steps,
                dt=self.dt,
                split=self.split,
                train_val_test=self.train_val_test,
                subname=self.subname,
                extra_specific=True,
            )
            sub_dsets.append(subd)
        return sub_dsets

    def _get_specific_stats(self, f: h5py.File) -> tuple[int, int]:
        raise NotImplementedError

    def _get_specific_bcs(self, f: h5py.File) -> list:
        raise NotImplementedError

    def _reconstruct_sample(
        self,
        file: h5py.File,
        sample_idx: int,
        time_idx: int,
        n_steps: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _get_directory_stats(self, path: str) -> None:
        self.files_paths = glob.glob(path + "/*.h5") + glob.glob(path + "/*.hdf5")
        self.files_paths.sort()
        self.n_files = len(self.files_paths)
        self.file_steps = []
        self.file_nsteps = []
        self.file_samples = []
        self.split_offsets = []
        self.offsets = [0]
        file_paths = []
        for file in self.files_paths:
            if len(self.include_string) > 0 and self.include_string not in file:
                continue
            elif file in broken_paths:
                continue
            file_paths.append(file)
            try:
                with h5py.File(file, "r") as _f:
                    samples, steps = self._get_specific_stats(_f)
                    if steps - self.n_steps - (self.dt - 1) < 1:
                        print(
                            f"WARNING: File {file} has {steps} steps, "
                            f"but n_steps is {self.n_steps}."
                            f"Setting file steps = max allowable."
                        )
                        file_nsteps = steps - self.dt
                    else:
                        file_nsteps = self.n_steps
                    self.file_nsteps.append(file_nsteps)
                    self.file_steps.append(steps - file_nsteps - (self.dt - 1))
                    if self.split_level == "sample":
                        partition = self.partition
                        sample_per_part = np.ceil(
                            np.array(self.train_val_test) * samples
                        ).astype(int)
                        sample_per_part[2] = max(
                            samples - sample_per_part[0] - sample_per_part[1], 0
                        )
                        self.split_offsets.append(
                            self.file_steps[-1] * sum(sample_per_part[:partition])
                        )
                        split_samples = sample_per_part[partition]
                    else:
                        split_samples = samples
                    self.file_samples.append(split_samples)
                    self.offsets.append(
                        self.offsets[-1]
                        + (steps - file_nsteps - (self.dt - 1)) * split_samples
                    )
            except Exception as e:
                print(f"WARNING: Failed to open file {file}. Continuing without it.")
                raise RuntimeError(f"Failed to open file {file}") from e

        self.files_paths = file_paths
        self.offsets[0] = -1
        self.files = [None for _ in self.files_paths]
        self.len = self.offsets[-1]

        if self.split_level == "file":
            if self.train_val_test is None:
                print(
                    "WARNING: No train/val/test split specified."
                    " Using all data for training."
                )
                self.split_offset = 0
                self.len = self.offsets[-1]
            else:
                print(f"Using train/val/test split: {self.train_val_test}")
                total_samples = sum(self.file_samples)
                ideal_split_offsets = [
                    int(self.train_val_test[i] * total_samples) for i in range(3)
                ]
                end_ind = 0
                for i in range(self.partition + 1):
                    run_sum = 0
                    start_ind = end_ind
                    for samples, steps in zip(self.file_samples, self.file_steps):
                        run_sum += samples
                        if run_sum <= ideal_split_offsets[i]:
                            end_ind += samples * steps
                            if run_sum == ideal_split_offsets[i]:
                                break
                        else:
                            delta = abs((run_sum - samples) - ideal_split_offsets[i])
                            end_ind += delta * steps
                            break
                self.split_offset = start_ind
                self.len = end_ind - start_ind

    def _open_file(self, file_ind: int) -> None:
        _file = h5py.File(self.files_paths[file_ind], "r")
        self.files[file_ind] = _file

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """No docstring."""
        if self.split_level == "file":
            index += self.split_offset

        file_idx = int(np.searchsorted(self.offsets, index, side="right") - 1)
        nsteps = self.file_nsteps[file_idx]
        local_idx = index - max(self.offsets[file_idx], 0)
        if self.split_level == "sample":
            sample_idx = (local_idx + self.split_offsets[file_idx]) // self.file_steps[
                file_idx
            ]
        else:
            sample_idx = local_idx // self.file_steps[file_idx]
        time_idx = local_idx % self.file_steps[file_idx]

        if self.files[file_idx] is None:
            self._open_file(file_idx)

        time_idx = (
            time_idx - self.dt if time_idx >= self.file_steps[file_idx] else time_idx
        )
        time_idx += nsteps
        try:
            trajectory = self._reconstruct_sample(
                self.files[file_idx], sample_idx, time_idx, nsteps
            )
            bcs = self._get_specific_bcs(self.files[file_idx])
        except Exception as e:
            raise RuntimeError(
                f"Failed to reconstruct sample for file {self.files_paths[file_idx]} "
                f"sample {sample_idx} time {time_idx}"
            ) from e

        return trajectory[:-1], torch.as_tensor(bcs), trajectory[-1]

    def __len__(self) -> int:
        """No docstring."""
        return self.len


class SWEDataset(BaseHDF5DirectoryDataset):
    """Dataset class for the 2D Shallow Water Equations (SWE).

    This dataset expects HDF5 files where each sample contains a time series of 2D scalar
    fields (e.g., fluid height) stored under the key 'data'. It uses a sample-based split
    strategy, with each sample identified by a group name in the HDF5 file.

    Fields:
        - 'h': fluid height

    Time/space structure:
        - Input sequence is extracted with a time window centered on `time_idx`, using
          `n_steps` and `dt` as history/depth parameters.
        - Output tensor is shaped (T, C, H, W), where:
            - T = number of time steps (history + future)
            - C = 1 channel (height field)
            - H, W = spatial dimensions
    """
    @staticmethod
    def _specifics() -> tuple[int, Optional[int], list[str], str, str]:
        """No docstring."""
        time_index = 0
        sample_index = None
        field_names = ["h"]
        type = "swe"
        split_level = "sample"
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f: h5py.File) -> tuple[int, int]:
        """No docstring."""
        samples = list(f.keys())
        steps = f[samples[0]]["data"].shape[0]
        return len(samples), steps

    def _get_specific_bcs(self, f: h5py.File) -> list[int]:
        """No docstring."""
        return [0, 0]  # Non-periodic

    def _reconstruct_sample(
        self, file: h5py.File, sample_idx: int, time_idx: int, n_steps: int
    ) -> np.ndarray:
        """No docstring."""
        samples = list(file.keys())
        return file[samples[sample_idx]]["data"][
            time_idx - n_steps * self.dt : time_idx + self.dt
        ].transpose(0, 3, 1, 2)


class DiffRe2DDataset(BaseHDF5DirectoryDataset):
    """No docstring."""
    @staticmethod
    def _specifics() -> tuple[int, None, list[str], str, str]:
        """No docstring."""
        time_index = 0
        sample_index = None
        field_names = ["activator", "inhibitor"]
        type = "diffre2d"
        split_level = "sample"
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f: h5py.File) -> tuple[int, int]:
        """No doctrsing."""
        samples = list(f.keys())
        steps = f[samples[0]]["data"].shape[0]
        return len(samples), steps

    def _get_specific_bcs(self, f: h5py.File) -> list[int]:
        """No doctrsing."""
        return [0, 0]  # Non-periodic

    def _reconstruct_sample(
        self, file: h5py.File, sample_idx: int, time_idx: int, n_steps: int
    ) -> np.ndarray:
        """No doctrsing."""
        samples = list(file.keys())
        return file[samples[sample_idx]]["data"][
            time_idx - n_steps * self.dt : time_idx + self.dt
        ].transpose(0, 3, 1, 2)


class IncompNSDataset(BaseHDF5DirectoryDataset):
    """Order Vx, Vy, "particles"."""

    @staticmethod
    def _specifics() -> tuple[int, int, list[str], str, str]:
        """No doctrsing."""
        time_index = 1
        sample_index = 0
        field_names = ["Vx", "Vy", "particles"]
        type = "incompNS"
        split_level = "file"
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f: h5py.File) -> tuple[int, int]:
        """No doctrsing."""
        samples = f["velocity"].shape[0]
        steps = f["velocity"].shape[1]  # Per dset
        return samples, steps

    def _reconstruct_sample(
        self, file: h5py.File, sample_idx: int, time_idx: int, n_steps: int
    ) -> np.ndarray:
        """No doctrsing."""
        velocity = file["velocity"][
            sample_idx, time_idx - n_steps * self.dt : time_idx + self.dt
        ]
        particles = file["particles"][
            sample_idx, time_idx - n_steps * self.dt : time_idx + self.dt
        ]
        comb = np.concatenate([velocity, particles], axis=-1)
        return comb.transpose((0, 3, 1, 2))

    def _get_specific_bcs(self, f: h5py.File) -> list[int]:
        """No doctrsing."""
        return [0, 0]  # Non-periodic


class PDEArenaINS(BaseHDF5DirectoryDataset):
    """Order Vx, Vy, density, pressure."""

    @staticmethod
    def _specifics() -> tuple[int, int, list[str], str, str]:
        """No doctrsing."""
        time_index = 1
        sample_index = 0
        field_names = ["Vx", "Vy", "u"]
        type = "pa_ins"
        split_level = "sample"
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f: h5py.File) -> tuple[int, int]:
        """No doctrsing."""
        samples = f["Vx"].shape[0]
        steps = f["Vx"].shape[1]  # Per dset
        return samples, steps

    def more_specific_title(self, type: str, path: str, include_string: str) -> str:
        """Override this to add more info to the dataset name."""
        split_path = self.include_string.split("/")[-1].split("_")
        buoy = split_path[-3]
        nu = split_path[-2]
        return f"{type}_buoy{buoy}_nu{nu}"

    def _reconstruct_sample(
        self, file: h5py.File, sample_idx: int, time_idx: int, n_steps: int
    ) -> np.ndarray:
        """No doctrsing."""
        vx = file["Vx"][sample_idx, time_idx - n_steps * self.dt : time_idx + self.dt]
        vy = file["Vy"][sample_idx, time_idx - n_steps * self.dt : time_idx + self.dt]
        density = file["u"][
            sample_idx, time_idx - n_steps * self.dt : time_idx + self.dt
        ]
        comb = np.stack([vx, vy, density], axis=1)
        return comb  # Already correct shape; no transpose needed

    def _get_specific_bcs(self, f: h5py.File) -> list[int]:
        """No doctrsing."""
        return [0, 0]  # Not Periodic


class CompNSDataset(BaseHDF5DirectoryDataset):
    """Order Vx, Vy, density, pressure."""

    @staticmethod
    def _specifics() -> tuple[int, int, list[str], str, str]:
        """No doctrsing."""
        time_index = 1
        sample_index = 0
        field_names = ["Vx", "Vy", "density", "pressure"]
        type = "compNS"
        split_level = "sample"
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f: h5py.File) -> tuple[int, int]:
        """No doctrsing."""
        samples = f["Vx"].shape[0]
        steps = f["Vx"].shape[1]  # Per dset
        return samples, steps

    def more_specific_title(self, type: str, path: str, include_string: str) -> str:
        """Override this to add more info to the dataset name."""
        cns_path = self.include_string.split("/")[-1].split("_")
        ic = cns_path[2]
        m = cns_path[3]
        res = cns_path[-2]
        return f"{type}_{ic}_{m}_res{res}"

    def _reconstruct_sample(
        self, file: h5py.File, sample_idx: int, time_idx: int, n_steps: int
    ) -> np.ndarray:
        """No doctrsing."""
        vx = file["Vx"][sample_idx, time_idx - n_steps * self.dt : time_idx + self.dt]
        vy = file["Vy"][sample_idx, time_idx - n_steps * self.dt : time_idx + self.dt]
        density = file["density"][
            sample_idx, time_idx - n_steps * self.dt : time_idx + self.dt
        ]
        p = file["pressure"][
            sample_idx, time_idx - n_steps * self.dt : time_idx + self.dt
        ]
        comb = np.stack([vx, vy, density, p], axis=1)
        return comb  # Already in correct format

    def _get_specific_bcs(self, f: h5py.File) -> list[int]:
        """No doctrsing."""
        return [1, 1]  # Periodic


class BurgersDataset(BaseHDF5DirectoryDataset):
    """Order Vx, Vy, density, pressure."""

    @staticmethod
    def _specifics() -> tuple[int, int, list[str], str, str]:
        """No doctrsing."""
        time_index = 1
        sample_index = 0
        field_names = ["Vx"]
        type = "burgers"
        split_level = "sample"
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f: h5py.File) -> tuple[int, int]:
        """No doctrsing."""
        samples = f["tensor"].shape[0]
        steps = f["tensor"].shape[1]  # Per dset
        return samples, steps

    def _reconstruct_sample(
        self, file: h5py.File, sample_idx: int, time_idx: int, n_steps: int
    ) -> np.ndarray:
        """No doctrsing."""
        vx = file["tensor"][
            sample_idx, time_idx - n_steps * self.dt : time_idx + self.dt
        ]
        vx = vx[:, None, :, None]
        return vx  # Already reshaped

    def _get_specific_bcs(self, f: h5py.File) -> list[int]:
        """No doctrsing."""
        return [1, 1]  # Periodic


class DiffSorb1DDataset(BaseHDF5DirectoryDataset):
    """No doctrsing."""
    @staticmethod
    def _specifics() -> tuple[int, None, list[str], str, str]:
        """No doctrsing."""
        time_index = 0
        sample_index = None
        field_names = ["u"]
        type = "diffsorb"
        split_level = "sample"
        return time_index, sample_index, field_names, type, split_level

    def _get_specific_stats(self, f: h5py.File) -> tuple[int, int]:
        samples = list(f.keys())
        steps = f[samples[0]]["data"].shape[0]
        return len(samples), steps

    def _get_specific_bcs(self, f: h5py.File) -> list[int]:
        return [0, 0]  # Non-periodic

    def _reconstruct_sample(
        self, file: h5py.File, sample_idx: int, time_idx: int, n_steps: int
    ) -> np.ndarray:
        samples = list(file.keys())
        return file[samples[sample_idx]]["data"][
            time_idx - n_steps * self.dt : time_idx + self.dt
        ].transpose(0, 2, 1)[:, :, :, None]
