"""Relating to kilonova light-curve data.

Functions and classes for torch DataSets which sample kilonova light-curve
(g/r/i band) data, along with helpers for computing and caching per-band
normalization statistics.

"""

####################################
# Packages
####################################
import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


EPS = 1e-6

# Nine-band ordering used by the merged event-stream dataset: the three ZTF
# bands followed by the six Rubin/LSST-style bands (u from SDSS, g/r/i/z/y from
# PanSTARRS naming). The index of each key in this tuple is the band index used
# throughout the 9-band pipeline.
NINE_BAND_KEYS = (
    "arr_ztfg",
    "arr_ztfr",
    "arr_ztfi",
    "arr_sdssu",
    "arr_ps1__g",
    "arr_ps1__r",
    "arr_ps1__i",
    "arr_ps1__z",
    "arr_ps1__y",
)


def compute_band_normalization(
    file_prefix_list: list[str],
    band_keys: tuple[str, ...] = ("arr_ztfg", "arr_ztfr", "arr_ztfi"),
    value_col: int = 1,
    error_col: int = 2,
    drop_upper_limits: bool = False,
    stats_path: str = "kilonova_gri_norm_stats.npz",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute global per-band mean/std over the training files only.

    Args:
        file_prefix_list (list[str]): List of npz files to accumulate stats over.
        band_keys (tuple[str, ...]): Keys of the bands to normalize.
        value_col (int): Column index of the value to accumulate per band.
        error_col (int): Column index of the per-observation uncertainty. Only
            used when drop_upper_limits is True.
        drop_upper_limits (bool): If True, observations with a non-finite
            uncertainty in error_col (upper limits / non-detections) are excluded
            from the statistics, matching a dataset that drops them.
        stats_path (str): Path to save the computed statistics to.

    Returns:
        means (np.ndarray): Per-band means, shape [n_bands].
        stds (np.ndarray): Per-band standard deviations, shape [n_bands].
    """
    sums = np.zeros(len(band_keys), dtype=np.float64)
    sums_sq = np.zeros(len(band_keys), dtype=np.float64)
    counts = np.zeros(len(band_keys), dtype=np.float64)

    for fn in file_prefix_list:
        data = np.load(fn, allow_pickle=True)

        for b, key in enumerate(band_keys):
            vals = data[key][:, value_col].astype(np.float64)

            finite = np.isfinite(vals)

            # Optionally exclude upper limits (non-finite uncertainty) so the
            # normalization statistics match a dataset that drops them.
            if drop_upper_limits:
                errs = data[key][:, error_col].astype(np.float64)
                finite = finite & np.isfinite(errs)

            vals = vals[finite]

            sums[b] += vals.sum()
            sums_sq[b] += np.square(vals).sum()
            counts[b] += vals.size

        data.close()

    means = sums / counts
    variances = sums_sq / counts - means**2
    variances = np.maximum(variances, 1e-12)
    stds = np.sqrt(variances)

    means = means.astype(np.float32)
    stds = stds.astype(np.float32)

    np.savez(
        stats_path,
        means=means,
        stds=stds,
        band_keys=np.array(band_keys),
        value_col=value_col,
    )

    print("Saved normalization stats:", stats_path)
    print("means:", means)
    print("stds:", stds)

    return means, stds


def load_or_compute_band_normalization(
    stats_path: str = "kilonova_gri_norm_stats.npz",
    band_keys: tuple[str, ...] = ("arr_ztfg", "arr_ztfr", "arr_ztfi"),
    value_col: int = 1,
    error_col: int = 2,
    drop_upper_limits: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load cached per-band normalization stats or compute them if missing.

    Args:
        stats_path (str): Path to load/save the statistics.
        band_keys (tuple[str, ...]): Keys of the bands to normalize.
        value_col (int): Column index of the value to accumulate per band.
        error_col (int): Column index of the per-observation uncertainty. Only
            used when drop_upper_limits is True.
        drop_upper_limits (bool): If True, exclude upper limits (non-finite
            uncertainty) from the statistics, matching a dataset that drops them.

    Returns:
        means (np.ndarray): Per-band means, shape [n_bands].
        stds (np.ndarray): Per-band standard deviations, shape [n_bands].
    """
    # FIXME: hardcoded scratch path. Should be passed in as an argument so this
    # library function does not depend on a user-specific filesystem location.
    file_prefix_list = sorted(
        glob.glob(
        "/net/sescratch1/atoivonen/data/KN_lightcurves/rubin_ztf_10000_dataset/lc_*.npz")
    )

    if os.path.exists(stats_path):
        stats = np.load(stats_path, allow_pickle=True)
        means = stats["means"].astype(np.float32)
        stds = stats["stds"].astype(np.float32)
        stats.close()

        print("Loaded normalization stats:", stats_path)
        print("means:", means)
        print("stds:", stds)

        return means, stds

    return compute_band_normalization(
        file_prefix_list=file_prefix_list,
        band_keys=band_keys,
        value_col=value_col,
        error_col=error_col,
        drop_upper_limits=drop_upper_limits,
        stats_path=stats_path,
    )


class Kilonova_lc_scalar_context_DataSet_gri(Dataset):
    """Scalar-context kilonova light-curve dataset for g/r/i bands.

    Each sample provides a flattened window of normalized per-band values plus
    relative observation times as input, and either the normalized next value
    or the normalized delta as the target.
    """

    def __init__(
        self,
        N_imgs: int = 0,
        context_len: int = 5,
        band_keys: tuple[str, ...] = ("arr_ztfg", "arr_ztfr", "arr_ztfi"),
        value_col: int = 1,
        means: np.ndarray = None,
        stds: np.ndarray = None,
        predicts_delta: bool = True,
    ) -> None:
        """Initialize the dataset and build the sample index.

        Args:
            N_imgs (int): Number of light-curve files to sample; 0 uses all.
            context_len (int): Number of context timesteps per sample.
            band_keys (tuple[str, ...]): Keys of the bands to load.
            value_col (int): Column index of the value to load per band.
            means (np.ndarray): Per-band means for normalization, shape [n_bands].
            stds (np.ndarray): Per-band stds for normalization, shape [n_bands].
            predicts_delta (bool): If True target is the normalized delta,
                otherwise the normalized next value.
        """
        # FIXME: hardcoded scratch path. Should be passed in as an argument so
        # this dataset does not depend on a user-specific filesystem location.
        file_prefix_list = sorted(
            glob.glob(
                "/net/sescratch1/atoivonen/data/KN_lightcurves/"
                "uniform_dataset_20000/lc_*.npz"
            )
        )

        if N_imgs == 0:
            self.file_prefix_list = file_prefix_list
        else:
            self.file_prefix_list = list(
                np.random.choice(file_prefix_list, N_imgs, replace=False)
            )

        random.shuffle(self.file_prefix_list)

        self.context_len = context_len
        self.band_keys = tuple(band_keys)
        self.value_col = value_col
        self.n_channels = len(self.band_keys)
        self.predicts_delta = predicts_delta

        if means is None:
            raise ValueError(
                "means must be provided for per-band normalization. "
                "Expected shape [n_channels], e.g. [g_mean, r_mean, i_mean]."
            )

        if stds is None:
            raise ValueError(
                "stds must be provided for per-band normalization. "
                "Expected shape [n_channels], e.g. [g_std, r_std, i_std]."
            )

        self.means = np.asarray(means, dtype=np.float32)
        self.stds = np.asarray(stds, dtype=np.float32)

        if self.means.shape[0] != self.n_channels:
            raise ValueError(
                f"means has length {self.means.shape[0]}, "
                f"but n_channels={self.n_channels}"
            )

        if self.stds.shape[0] != self.n_channels:
            raise ValueError(
                f"stds has length {self.stds.shape[0]}, "
                f"but n_channels={self.n_channels}"
            )

        if np.any(self.stds <= 0):
            raise ValueError(f"All stds must be positive. Got stds={self.stds}")

        self.samples = []

        for file_idx, fn in enumerate(self.file_prefix_list):
            data = np.load(fn, allow_pickle=True)

            # Assume all bands share the same time grid.
            mjd = data[self.band_keys[0]][:, 0]
            n_times = len(mjd)

            data.close()

            max_start = n_times - context_len - 1
            for startIDX in range(max_start + 1):
                self.samples.append((file_idx, startIDX))

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the (input, target, Dt) triple for a given sample index.

        Args:
            index (int): Sample index.

        Returns:
            x (torch.Tensor): Flattened context values and relative times.
            target (torch.Tensor): Normalized next value or delta per band.
            Dt (torch.Tensor): Time delta to the target step.
        """
        file_idx, startIDX = self.samples[index]
        fn = self.file_prefix_list[file_idx]

        data = np.load(fn, allow_pickle=True)

        # Time grid from the first band.
        # Assumes arr_ztfg, arr_ztfr, arr_ztfi have matching MJD columns.
        mjd = data[self.band_keys[0]][:, 0].astype(np.float32)

        # Stack one scalar value column from each band.
        # vals shape: [T, 3] for g/r/i.
        vals = np.stack(
            [
                data[key][:, self.value_col].astype(np.float32)
                for key in self.band_keys
            ],
            axis=1,
        )

        data.close()

        # Per-band normalization.
        # vals[:, 0] = normalized g
        # vals[:, 1] = normalized r
        # vals[:, 2] = normalized i
        vals = (vals - self.means[None, :]) / (self.stds[None, :] + EPS)

        t0 = mjd.min()
        t_obs = mjd - t0

        target_idx = startIDX + self.context_len

        # Context values shape: [context_len, 3]
        context_vals = vals[startIDX:target_idx]

        # Relative observation times shape: [context_len]
        rel_t = t_obs[startIDX:target_idx] - t_obs[startIDX]
        rel_t = rel_t.astype(np.float32)

        # Flatten context as:
        # [g0, r0, i0, g1, r1, i1, ..., gK, rK, iK]
        context_flat = context_vals.reshape(-1).astype(np.float32)

        # Final input shape:
        # [(3 * context_len) + context_len]
        #
        # For context_len=5:
        # x.shape == [20]
        x = np.concatenate([context_flat, rel_t], axis=0)
        x = torch.tensor(x, dtype=torch.float32)

        if self.predicts_delta:
            # Predict normalized delta for each band:
            # [delta_g, delta_r, delta_i]
            target_vals = vals[target_idx] - vals[target_idx - 1]
        else:
            # Predict normalized absolute next value:
            # [g_next, r_next, i_next]
            target_vals = vals[target_idx]

        target = torch.tensor(target_vals, dtype=torch.float32)

        Dt = torch.tensor(
            t_obs[target_idx] - t_obs[target_idx - 1],
            dtype=torch.float32,
        )

        return x, target, Dt


class Kilonova_lc_scalar_context_DataSet_9band(Dataset):
    """Merged event-stream kilonova dataset for sparse, irregular 9-band data.

    Unlike the g/r/i dataset, this class makes no assumption that the bands
    share a common time grid. Real and Rubin-simulated light curves record each
    band on its own cadence, with differing numbers of observations per band and
    entire bands sometimes missing. Every observation across all bands is read
    as-is, tagged with its band index, and merged into a single time-sorted
    stream of events.

    Each sample is a window of ``context_len`` consecutive events used to predict
    the next event in the stream. Because any single future observation belongs
    to one band, the target is stored as a length-``n_bands`` vector with a mask
    selecting the observed band; the loss is applied only there. Across the whole
    dataset every band is supervised, and at inference the model emits a
    prediction for all bands at a chosen lead time.

    Each returned sample is a ``(x, target, mask, Dt)`` tuple where:
        x:      flattened context of shape
                [context_len * (2 + n_bands)], laid out per event as
                [value, rel_t, one_hot_band(n_bands)].
        target: normalized value per band, shape [n_bands]; only the observed
                band is meaningful.
        mask:   float mask, shape [n_bands]; 1.0 for the observed target band,
                0.0 elsewhere.
        Dt:     lead time from the last context event to the target event.
    """

    def __init__(
        self,
        N_imgs: int = 0,
        context_len: int = 5,
        band_keys: tuple[str, ...] = NINE_BAND_KEYS,
        value_col: int = 1,
        error_col: int = 2,
        drop_upper_limits: bool = True,
        means: np.ndarray = None,
        stds: np.ndarray = None,
        n_rollout_steps: int = 1,
    ) -> None:
        """Initialize the dataset and build the merged-event sample index.

        Args:
            N_imgs (int): Number of light-curve files to sample; 0 uses all.
            context_len (int): Number of context events per sample.
            band_keys (tuple[str, ...]): Keys of the bands to load. Their order
                defines the band index used in the one-hot encoding and target.
            value_col (int): Column index of the value to load per band.
            error_col (int): Column index of the per-observation uncertainty.
                Upper-limit (non-detection) rows are flagged by a non-finite
                (e.g. inf) uncertainty in this column.
            drop_upper_limits (bool): If True, observations with a non-finite
                uncertainty in ``error_col`` are dropped from the event stream so
                the model only sees real detections.
            means (np.ndarray): Per-band means for normalization, shape [n_bands].
            stds (np.ndarray): Per-band stds for normalization, shape [n_bands].
            n_rollout_steps (int): Number of future events supervised per sample.
                When 1 (default) ``__getitem__`` returns the single-step
                ``(x, target, mask, Dt)`` tuple. When >1 it returns a rollout
                tuple carrying the initial context window plus the next
                ``n_rollout_steps`` true events, for scheduled-sampling /
                multi-step rollout training (see ``__getitem__``).
        """
        # FIXME: hardcoded scratch path. Should be passed in as an argument so
        # this dataset does not depend on a user-specific filesystem location.
        # NOTE: must point at the SAME dataset as
        # load_or_compute_band_normalization (the rubin_ztf_10000 set). The old
        # uniform_dataset_20000 set is ZTF-only, so training on it left the six
        # Rubin output heads without any targets (never trained) while the norm
        # stats were computed over Rubin+ZTF -- a silent train/stats mismatch.
        file_prefix_list = sorted(
            glob.glob(
                "/net/sescratch1/atoivonen/data/KN_lightcurves/"
                "rubin_ztf_10000_dataset/lc_*.npz"
            )
        )

        if N_imgs == 0:
            self.file_prefix_list = file_prefix_list
        else:
            self.file_prefix_list = list(
                np.random.choice(file_prefix_list, N_imgs, replace=False)
            )

        random.shuffle(self.file_prefix_list)

        self.context_len = context_len
        self.band_keys = tuple(band_keys)
        self.value_col = value_col
        self.error_col = error_col
        self.drop_upper_limits = drop_upper_limits
        self.n_channels = len(self.band_keys)

        if n_rollout_steps < 1:
            raise ValueError(
                f"n_rollout_steps must be >= 1, got {n_rollout_steps}"
            )
        self.n_rollout_steps = n_rollout_steps

        if means is None:
            raise ValueError(
                "means must be provided for per-band normalization. "
                "Expected shape [n_channels]."
            )

        if stds is None:
            raise ValueError(
                "stds must be provided for per-band normalization. "
                "Expected shape [n_channels]."
            )

        self.means = np.asarray(means, dtype=np.float32)
        self.stds = np.asarray(stds, dtype=np.float32)

        if self.means.shape[0] != self.n_channels:
            raise ValueError(
                f"means has length {self.means.shape[0]}, "
                f"but n_channels={self.n_channels}"
            )

        if self.stds.shape[0] != self.n_channels:
            raise ValueError(
                f"stds has length {self.stds.shape[0]}, "
                f"but n_channels={self.n_channels}"
            )

        if np.any(self.stds <= 0):
            raise ValueError(f"All stds must be positive. Got stds={self.stds}")

        # Build the merged, time-sorted event stream for each file and index
        # every context window into it. Events are (rel_time, value, band_idx).
        # rel_time is relative to the earliest observation across all bands in
        # the file so absolute MJD offsets do not leak into the model.
        self.events_per_file = []
        self.samples = []

        for file_idx, fn in enumerate(self.file_prefix_list):
            data = np.load(fn, allow_pickle=True)

            times = []
            values = []
            bands = []

            for band_idx, key in enumerate(self.band_keys):
                # A band may be missing entirely in a given file.
                if key not in data.files:
                    continue

                arr = data[key]
                if arr.size == 0:
                    continue

                # Drop upper-limit (non-detection) rows, which are flagged by a
                # non-finite uncertainty (e.g. inf) in error_col. This keeps only
                # real detections in the merged event stream.
                if self.drop_upper_limits:
                    detected = np.isfinite(arr[:, self.error_col])
                    arr = arr[detected]
                    if arr.shape[0] == 0:
                        continue

                times.append(arr[:, 0].astype(np.float32))
                values.append(arr[:, value_col].astype(np.float32))
                bands.append(np.full(arr.shape[0], band_idx, dtype=np.int64))

            data.close()

            if not times:
                continue

            times = np.concatenate(times)
            values = np.concatenate(values)
            bands = np.concatenate(bands)

            # Sort the merged stream by observation time.
            order = np.argsort(times, kind="stable")
            times = times[order]
            values = values[order]
            bands = bands[order]

            # Relative times within the file.
            times = times - times.min()

            # Per-band normalization of the values.
            values = (values - self.means[bands]) / (self.stds[bands] + EPS)

            self.events_per_file.append(
                (times, values.astype(np.float32), bands)
            )

            n_events = times.shape[0]
            max_start = n_events - context_len - 1
            for startIDX in range(max_start + 1):
                self.samples.append((len(self.events_per_file) - 1, startIDX))

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        """Return the sample for ``index``.

        When ``n_rollout_steps == 1`` this returns the single-step
        ``(x, target, mask, Dt)`` tuple. When ``n_rollout_steps > 1`` it returns
        the rollout tuple ``(ctx_v, ctx_t, ctx_b, future_v, future_b, future_dt,
        future_valid)`` described in :meth:`_getitem_rollout`.

        Args:
            index (int): Sample index.

        Returns:
            tuple[torch.Tensor, ...]: Single-step or rollout sample.
        """
        if self.n_rollout_steps > 1:
            return self._getitem_rollout(index)

        return self._getitem_single(index)

    def _getitem_single(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the (input, target, mask, Dt) tuple for a given sample index.

        Args:
            index (int): Sample index.

        Returns:
            x (torch.Tensor): Flattened per-event context of shape
                [context_len * (2 + n_bands)].
            target (torch.Tensor): Normalized value per band, shape [n_bands];
                only the observed band is meaningful.
            mask (torch.Tensor): Float mask, shape [n_bands]; 1.0 for the
                observed target band, 0.0 elsewhere.
            Dt (torch.Tensor): Lead time to the target event.
        """
        file_idx, startIDX = self.samples[index]
        times, values, bands = self.events_per_file[file_idx]

        target_idx = startIDX + self.context_len

        # Context events.
        ctx_t = times[startIDX:target_idx]
        ctx_v = values[startIDX:target_idx]
        ctx_b = bands[startIDX:target_idx]

        # Relative times within the context window.
        rel_t = (ctx_t - ctx_t[0]).astype(np.float32)

        # One-hot encode the band of each context event.
        band_onehot = np.zeros(
            (self.context_len, self.n_channels), dtype=np.float32
        )
        band_onehot[np.arange(self.context_len), ctx_b] = 1.0

        # Per-event feature: [value, rel_t, one_hot_band(n_bands)].
        per_event = np.concatenate(
            [
                ctx_v[:, None],
                rel_t[:, None],
                band_onehot,
            ],
            axis=1,
        )

        # Flatten to [context_len * (2 + n_bands)].
        x = torch.tensor(per_event.reshape(-1), dtype=torch.float32)

        # Target is the next event, placed into a per-band vector with a mask
        # marking which band was actually observed.
        target_band = int(bands[target_idx])
        target = np.zeros(self.n_channels, dtype=np.float32)
        mask = np.zeros(self.n_channels, dtype=np.float32)
        target[target_band] = values[target_idx]
        mask[target_band] = 1.0

        target = torch.tensor(target, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        Dt = torch.tensor(
            times[target_idx] - times[target_idx - 1],
            dtype=torch.float32,
        )

        return x, target, mask, Dt

    def _getitem_rollout(
        self, index: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return a multi-step rollout sample for scheduled-sampling training.

        The initial context window is returned in a structured form so the
        training loop can rebuild the model input after feeding predictions back
        in, and the next ``n_rollout_steps`` true events are returned as
        fixed-length arrays. When the stream ends before ``n_rollout_steps``
        future events are available, the remaining steps are padded and flagged
        invalid via ``future_valid`` so the default collate can still stack
        samples and the loss can ignore the padding.

        Args:
            index (int): Sample index.

        Returns:
            ctx_v (torch.Tensor): Normalized context values, shape [context_len].
            ctx_t (torch.Tensor): Relative context times (relative to the first
                context event), shape [context_len].
            ctx_b (torch.Tensor): Context band indices (long), shape [context_len].
            future_v (torch.Tensor): Normalized true value of each future event,
                shape [n_rollout_steps]; padded steps are 0.
            future_b (torch.Tensor): Band index of each future event (long),
                shape [n_rollout_steps]; padded steps are 0.
            future_dt (torch.Tensor): Lead time from the previous event to each
                future event, shape [n_rollout_steps]; padded steps are 0.
            future_valid (torch.Tensor): 1.0 for real future events, 0.0 for
                padded steps, shape [n_rollout_steps].
        """
        file_idx, startIDX = self.samples[index]
        times, values, bands = self.events_per_file[file_idx]

        target_start = startIDX + self.context_len

        # Initial context window, relative to its own first event so no absolute
        # MJD offset leaks in (matching the single-step encoding).
        ctx_t = (
            times[startIDX:target_start] - times[startIDX]
        ).astype(np.float32)
        ctx_v = values[startIDX:target_start].astype(np.float32)
        ctx_b = bands[startIDX:target_start].astype(np.int64)

        n = self.n_rollout_steps
        future_v = np.zeros(n, dtype=np.float32)
        future_b = np.zeros(n, dtype=np.int64)
        future_dt = np.zeros(n, dtype=np.float32)
        future_valid = np.zeros(n, dtype=np.float32)

        n_events = times.shape[0]
        for step in range(n):
            target_idx = target_start + step
            if target_idx >= n_events:
                break

            future_v[step] = values[target_idx]
            future_b[step] = bands[target_idx]
            future_dt[step] = times[target_idx] - times[target_idx - 1]
            future_valid[step] = 1.0

        return (
            torch.tensor(ctx_v, dtype=torch.float32),
            torch.tensor(ctx_t, dtype=torch.float32),
            torch.tensor(ctx_b, dtype=torch.long),
            torch.tensor(future_v, dtype=torch.float32),
            torch.tensor(future_b, dtype=torch.long),
            torch.tensor(future_dt, dtype=torch.float32),
            torch.tensor(future_valid, dtype=torch.float32),
        )
