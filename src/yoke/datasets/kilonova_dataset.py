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


def _stem(path: str) -> str:
    """Return the object identifier (filename without directory or extension).

    Light-curve files are named ``lc_<id>.npz``; the stem ``lc_<id>`` is the
    object identity used to pair the same object across the realistic and dense
    directories and to build object-level train/val/test splits. ``.npz`` is a
    single extension so ``splitext`` is exact.
    """
    return os.path.splitext(os.path.basename(path))[0]


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
    file_prefix_list: list[str] = None,
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
        file_prefix_list (list[str]): Files to accumulate stats over. Pass the
            TRAIN files only to avoid val/test leakage. If None, falls back to
            the legacy hardcoded scratch glob (kept for backward compatibility).

    Returns:
        means (np.ndarray): Per-band means, shape [n_bands].
        stds (np.ndarray): Per-band standard deviations, shape [n_bands].
    """
    if file_prefix_list is None:
        # FIXME: hardcoded scratch path fallback. Prefer passing an explicit
        # (train-only) file_prefix_list so this library function does not depend
        # on a user-specific filesystem location and does not leak val/test data.
        file_prefix_list = sorted(
            glob.glob(
                "/net/sescratch1/atoivonen/data/KN_lightcurves/"
                "rubin_ztf_10000_dataset_same_seed/lc_*.npz"
            )
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
        context_window_days: float = None,
        max_context_len: int = None,
        target_horizon_days: float = None,
        data_glob: str = None,
        object_ids: set = None,
    ) -> None:
        """Initialize the dataset and build the merged-event sample index.

        Args:
            N_imgs (int): Number of light-curve files to sample; 0 uses all.
            context_len (int): Number of context events per sample. In the
                default fixed-count mode this is the exact context length. It is
                ignored when ``context_window_days`` is set (window mode uses
                ``max_context_len`` for the padded width instead).
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
                multi-step rollout training (see ``__getitem__``). Only supported
                in fixed-count mode (``context_window_days`` is None).
            context_window_days (float): If set, switches to **time-window
                mode**: the context of each sample is every detection within this
                many days before the target event, padded to ``max_context_len``
                with a per-event validity flag (see ``_getitem_window``). If None
                (default) the dataset uses the legacy fixed-count context of
                exactly ``context_len`` events.
            max_context_len (int): Padded context width used in time-window mode.
                Windows with more real events than this keep only the most recent
                ``max_context_len``; windows with fewer are zero-padded. Defaults
                to ``context_len`` when not given. Unused in fixed-count mode.
            target_horizon_days (float): If set (window mode only), enables
                **horizon-covering target sampling**: instead of always
                supervising the immediate next event, each sample draws a target
                lead time ~uniform in days over ``(0, target_horizon_days]`` and
                supervises the event whose gap from the anchor is nearest that
                lead time (clamped to the last event in the curve). This flattens
                the supervised-``Delta_t`` distribution so the model is trained at
                the lead times it is later asked to forecast, rather than only at
                the short gap-to-next-event (median ~0.3d, p99 ~4d) while
                diagnostics forecast out to ~12d. The context selection (trailing
                ``context_window_days`` window ending at the anchor) is unchanged,
                preserving train/inference parity; only the target moves. In
                rollout mode each of the ``n_rollout_steps`` steps draws its own
                farther target from its current anchor, spreading coverage across
                the whole rollout. None (default) keeps the immediate-next-event
                target. Ignored in fixed-count mode.
            data_glob (str): Glob pattern selecting the light-curve files (i.e.
                which dataset directory). If None (default) the legacy hardcoded
                rubin_ztf_10000 scratch glob is used, preserving prior behavior.
                Used to point at either the realistic or the dense directory.
            object_ids (set): If given, only files whose stem (filename without
                directory or ``.npz``) is in this set are loaded. Used to apply a
                shared object-level train/val/test split across the realistic and
                dense directories (the same stems appear in both). None (default)
                loads all files matched by ``data_glob``.
        """
        # Select the dataset directory. NOTE: the chosen set must be consistent
        # with the normalization stats (both Rubin+ZTF). The old
        # uniform_dataset_20000 set is ZTF-only, so training on it left the six
        # Rubin output heads without any targets (never trained) while the norm
        # stats were computed over Rubin+ZTF -- a silent train/stats mismatch.
        if data_glob is None:
            # Legacy hardcoded scratch fallback (backward compatibility).
            data_glob = (
                "/net/sescratch1/atoivonen/data/KN_lightcurves/"
                "rubin_ztf_10000_dataset_same_seed/lc_*.npz"
            )
        file_prefix_list = sorted(glob.glob(data_glob))

        # Restrict to an object-level split (shared across the realistic and
        # dense directories via matching filename stems) when object_ids is set.
        if object_ids is not None:
            object_ids = set(object_ids)
            file_prefix_list = [
                f for f in file_prefix_list if _stem(f) in object_ids
            ]

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

        # Time-window context mode. When context_window_days is set, the context
        # is selected by a trailing lookback in days and padded to
        # max_context_len with a validity flag, instead of a fixed event count.
        self.context_window_days = context_window_days
        self.window_mode = context_window_days is not None

        if self.window_mode:
            self.max_context_len = (
                max_context_len if max_context_len is not None else context_len
            )
            if self.max_context_len < 1:
                raise ValueError(
                    f"max_context_len must be >= 1, got {self.max_context_len}"
                )
            if context_window_days <= 0:
                raise ValueError(
                    "context_window_days must be positive, got "
                    f"{context_window_days}"
                )
        else:
            self.max_context_len = context_len

        # Horizon-covering target sampling (window mode only). None keeps the
        # immediate-next-event target; a positive value draws a target lead time
        # ~uniform in days over (0, target_horizon_days].
        self.target_horizon_days = target_horizon_days
        if target_horizon_days is not None:
            if not self.window_mode:
                raise ValueError(
                    "target_horizon_days is only supported in time-window mode "
                    "(set context_window_days)."
                )
            if target_horizon_days <= 0:
                raise ValueError(
                    "target_horizon_days must be positive, got "
                    f"{target_horizon_days}"
                )

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
        # Object stem (identity) parallel to events_per_file. Needed because a
        # file's index in events_per_file is NOT its index in file_prefix_list
        # (files are shuffled above, and empty curves are skipped below), so
        # pairing the same object across the realistic and dense datasets must go
        # through the stem, not a positional index.
        self.stems_per_file = []
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
            self.stems_per_file.append(_stem(fn))

            n_events = times.shape[0]
            file_idx = len(self.events_per_file) - 1

            if self.window_mode:
                # One sample per event after the first: the target is event
                # target_idx and the context is the trailing window ending at
                # target_idx - 1 (always non-empty, so short curves contribute).
                for target_idx in range(1, n_events):
                    self.samples.append((file_idx, target_idx))
            else:
                # Legacy fixed-count windows: startIDX indexes the window start,
                # target is startIDX + context_len.
                max_start = n_events - context_len - 1
                for startIDX in range(max_start + 1):
                    self.samples.append((file_idx, startIDX))

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
        if self.window_mode:
            if self.n_rollout_steps > 1:
                return self._getitem_window_rollout(index)
            return self._getitem_window(index)

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

    def _draw_target_idx(
        self, times: np.ndarray, anchor_idx: int, n_events: int
    ) -> int:
        """Draw a horizon-covering target event index ahead of ``anchor_idx``.

        Samples a target lead time ~uniform in days over
        ``(0, target_horizon_days]`` and returns the future event whose gap from
        the anchor is nearest that lead time. Because reachable lead times are
        discrete (the actual future observation times), this approximates a
        uniform-in-days target distribution up to data availability, and clamps
        to the last event when the drawn lead time exceeds the curve. Assumes at
        least one event follows the anchor (``anchor_idx + 1 < n_events``).

        Args:
            times (np.ndarray): Merged, sorted, file-relative event times.
            anchor_idx (int): Index of the most recent context event.
            n_events (int): Number of events in the curve.

        Returns:
            int: The drawn target event index, in ``(anchor_idx, n_events)``.
        """
        lead_time = np.random.uniform(0.0, self.target_horizon_days)
        cand = np.arange(anchor_idx + 1, n_events)
        gaps = times[cand] - times[anchor_idx]
        return int(cand[np.argmin(np.abs(gaps - lead_time))])

    def _getitem_window(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the (input, target, mask, Dt) tuple in time-window mode.

        The context is every detection within ``context_window_days`` before the
        target event, padded to ``max_context_len`` with a per-event validity
        flag. The per-event feature gains a ``valid`` channel so the flattened
        input carries the padding mask itself (the model does no masking):
        ``[value, rel_t, valid, one_hot_band(n_bands)]``, width ``3 + n_bands``.

        Real events fill the leading rows in time order (oldest first, matching
        the fixed-count layout), with ``rel_t`` relative to the first *real*
        event in the window; padded rows are all-zero with ``valid = 0``.

        Args:
            index (int): Sample index (maps to a (file_idx, target_idx) pair).

        Returns:
            x (torch.Tensor): Flattened padded context, shape
                [max_context_len * (3 + n_bands)].
            target (torch.Tensor): Normalized value per band, shape [n_bands];
                only the observed band is meaningful.
            mask (torch.Tensor): Float mask, shape [n_bands]; 1.0 for the
                observed target band, 0.0 elsewhere.
            Dt (torch.Tensor): Lead time from the most recent context event to
                the target event.
        """
        file_idx, target_idx = self.samples[index]
        times, values, bands = self.events_per_file[file_idx]

        # Anchor the trailing window on the event immediately before the enumerated
        # target (the most recent observation). With horizon-covering target
        # sampling the supervised target is redrawn to a farther event so the
        # lead time Dt is ~uniform in days; the anchor (hence the context) is
        # unchanged, preserving train/inference parity.
        anchor_idx = target_idx - 1
        if self.target_horizon_days is not None:
            target_idx = self._draw_target_idx(
                times, anchor_idx, times.shape[0]
            )

        anchor_t = times[anchor_idx]
        lo = anchor_t - self.context_window_days

        # Context is events up to and including the anchor (never the target,
        # which may now be several events ahead) within the trailing window.
        prior_t = times[: anchor_idx + 1]
        in_window = prior_t >= lo
        sel_idx = np.nonzero(in_window)[0]
        if sel_idx.shape[0] > self.max_context_len:
            sel_idx = sel_idx[-self.max_context_len:]

        ctx_t = times[sel_idx]
        ctx_v = values[sel_idx]
        ctx_b = bands[sel_idx]
        n_real = sel_idx.shape[0]

        # rel_t relative to the first real event in the window (same convention
        # as the fixed-count path, which uses the window's first event).
        rel_t = (ctx_t - ctx_t[0]).astype(np.float32)

        # Padded per-event array: [value, rel_t, valid, one_hot_band].
        per_event = np.zeros(
            (self.max_context_len, 3 + self.n_channels), dtype=np.float32
        )
        per_event[:n_real, 0] = ctx_v
        per_event[:n_real, 1] = rel_t
        per_event[:n_real, 2] = 1.0  # validity flag for real events
        per_event[np.arange(n_real), 3 + ctx_b] = 1.0

        x = torch.tensor(per_event.reshape(-1), dtype=torch.float32)

        # Target is the event at target_idx, in a per-band vector + mask.
        target_band = int(bands[target_idx])
        target = np.zeros(self.n_channels, dtype=np.float32)
        mask = np.zeros(self.n_channels, dtype=np.float32)
        target[target_band] = values[target_idx]
        mask[target_band] = 1.0

        target = torch.tensor(target, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        # Lead time from the anchor (most recent context event) to the target,
        # which may be several events ahead under horizon-covering sampling.
        Dt = torch.tensor(
            times[target_idx] - times[anchor_idx],
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

    def _getitem_window_rollout(
        self, index: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return a multi-step rollout sample in time-window context mode.

        Combines the time-window context selection of :meth:`_getitem_window`
        with the multi-step future supervision of :meth:`_getitem_rollout`, so
        scheduled-sampling rollout training uses the SAME trailing-W-day context
        the model sees at inference (``get_rollout_from_stream`` /
        ``_select_window`` in ``plot_pred_diagnostics_9band.py``).

        The seed context is the trailing ``context_window_days`` window ending at
        the anchor event (``target_idx - 1``), capped to the most recent
        ``max_context_len`` events, and returned **padded** to ``max_context_len``
        with a per-event validity flag. Unlike :meth:`_getitem_rollout`, the
        context times are returned as **absolute stream times** (not made
        relative here) so the training loop can append the true future event
        times and re-select the W-day window each step, then make times relative
        per step exactly as inference does.

        Args:
            index (int): Sample index (maps to a (file_idx, target_idx) pair).

        Returns:
            ctx_v (torch.Tensor): Padded context values, shape [max_context_len];
                padded rows are 0.
            ctx_t (torch.Tensor): Padded ABSOLUTE context times, shape
                [max_context_len]; padded rows are 0.
            ctx_b (torch.Tensor): Padded context band indices (long), shape
                [max_context_len]; padded rows are 0.
            ctx_valid (torch.Tensor): 1.0 for real seed events, 0.0 for padding,
                shape [max_context_len].
            future_v (torch.Tensor): Normalized true value of each future event,
                shape [n_rollout_steps]; padded steps are 0.
            future_b (torch.Tensor): Band index of each future event (long),
                shape [n_rollout_steps]; padded steps are 0.
            future_dt (torch.Tensor): Lead time from the previous event to each
                future event, shape [n_rollout_steps]; padded steps are 0.
            future_valid (torch.Tensor): 1.0 for real future events, 0.0 for
                padded steps, shape [n_rollout_steps].
        """
        file_idx, target_idx = self.samples[index]
        times, values, bands = self.events_per_file[file_idx]

        # Seed context: trailing W-day window ending at the anchor event
        # (target_idx - 1), capped to the most recent max_context_len. Identical
        # selection to _getitem_window.
        anchor_t = times[target_idx - 1]
        lo = anchor_t - self.context_window_days

        prior_t = times[:target_idx]
        in_window = prior_t >= lo
        sel_idx = np.nonzero(in_window)[0]
        if sel_idx.shape[0] > self.max_context_len:
            sel_idx = sel_idx[-self.max_context_len:]

        n_real = sel_idx.shape[0]

        # Padded seed context; times kept ABSOLUTE so the loop can append true
        # future times and re-window (the loop makes them relative per step).
        ctx_v = np.zeros(self.max_context_len, dtype=np.float32)
        ctx_t = np.zeros(self.max_context_len, dtype=np.float32)
        ctx_b = np.zeros(self.max_context_len, dtype=np.int64)
        ctx_valid = np.zeros(self.max_context_len, dtype=np.float32)

        ctx_v[:n_real] = values[sel_idx]
        ctx_t[:n_real] = times[sel_idx]
        ctx_b[:n_real] = bands[sel_idx]
        ctx_valid[:n_real] = 1.0

        # Future events target_idx … target_idx + n - 1, padded past the stream
        # end and flagged invalid (same tail handling as _getitem_rollout).
        n = self.n_rollout_steps
        future_v = np.zeros(n, dtype=np.float32)
        future_b = np.zeros(n, dtype=np.int64)
        future_dt = np.zeros(n, dtype=np.float32)
        future_valid = np.zeros(n, dtype=np.float32)

        # Future targets. Without horizon sampling these are the consecutive
        # events target_idx … target_idx + n - 1 (immediate-next chain). With
        # horizon sampling each step draws a farther target from its own anchor
        # (the previous step's target), so the rollout is supervised at
        # ~uniform-in-days lead times at every step; future_dt is the anchor→
        # target gap the training loop uses to advance the growing buffer.
        n_events = times.shape[0]
        prev_idx = target_idx - 1
        for step in range(n):
            if prev_idx + 1 >= n_events:
                break

            if self.target_horizon_days is not None:
                t_idx = self._draw_target_idx(times, prev_idx, n_events)
            else:
                t_idx = prev_idx + 1

            future_v[step] = values[t_idx]
            future_b[step] = bands[t_idx]
            future_dt[step] = times[t_idx] - times[prev_idx]
            future_valid[step] = 1.0
            prev_idx = t_idx

        return (
            torch.tensor(ctx_v, dtype=torch.float32),
            torch.tensor(ctx_t, dtype=torch.float32),
            torch.tensor(ctx_b, dtype=torch.long),
            torch.tensor(ctx_valid, dtype=torch.float32),
            torch.tensor(future_v, dtype=torch.float32),
            torch.tensor(future_b, dtype=torch.long),
            torch.tensor(future_dt, dtype=torch.float32),
            torch.tensor(future_valid, dtype=torch.float32),
        )
