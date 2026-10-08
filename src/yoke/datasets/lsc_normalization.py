"""Per-channel normalization scheme for LSC hydrodynamic fields.

Implements the normalization logic used by LodeRunner-ViT LSC training (the
``ch_ldrViT`` 2-frame harnesses). This module owns the *scheme itself*
(classification, transforms, stats computation, and NPZ save/load/apply);
the file-sampling and NPZ-scanning orchestration used to *estimate* these
statistics across a dataset lives in
``applications/normalization/generate_lsc_channel_normalization.py``, which
imports from here.

This is deliberately independent of
:class:`~yoke.datasets.lsc_dataset.LSCnorm_cntr2rho_DataSet`'s normalization
(per-time-bin average-density unbiasing + B-spline design-parameter
normalization): that targets an unrelated geometry-to-final-state problem.
This module targets LodeRunner-ViT's variable-channel image-to-image
prediction (see
``misc_work_notes/ArtIMich/LodeRunnerViT_2Frame_improvement_plan.md`` and the
channel-normalization discussion that followed it).

Normalization scheme (all transforms are pure *scale*, never a *shift*, so a
physically-zero/ambient value stays exactly ``0`` after normalization):

- **Density fields** (``density_*``, i.e. ``volfrac * density``): one-sided.
  These fields are exactly ``0`` outside the material's footprint, so a shift
  would be both unnecessary and would destroy that sparsity. The scale factor
  is a high percentile (default 99.9th) of the field restricted to **nonzero**
  voxels, pooled over many sampled frames/simulations. Using a percentile
  rather than the literal max avoids a single outlier voxel/simulation setting
  the scale for the whole corpus.
- **Velocity fields** (``Uvelocity``, ``Wvelocity``): two-sided, signed, not
  log-transformed (velocities are not inherently log-range). The scale factor
  is ``max(|p_low|, |p_high|)`` where ``p_low``/``p_high`` are the (default)
  0.1/99.9 percentiles over all voxels (zero included), pooled over sampled
  frames. This is a *symmetric* scale so ``v=0`` stays ``0`` and sign is
  preserved.
- **Pressure/energy fields** (``pressure_*``, ``energy_*``): two-sided and
  log-range (pressure is the trace of the stress tensor and can go negative).
  Normalized via a signed-log1p transform, ``sign(x) * log1p(|x|)``, followed
  by a symmetric percentile scale identical in spirit to the velocity case but
  applied to the log1p-transformed values. Included for forward-compatibility
  with a future 20-channel study; not exercised by the current 8-channel
  harness.
"""

import numpy as np

#: Fields reweighted by their volume-fraction companion field (see
#: :func:`yoke.datasets.lsc_dataset.volfrac_density`).
DENSITY_PREFIX = "density_"
#: Fields treated as signed, log-range quantities (pressure, energy).
LOG_RANGE_PREFIXES = ("pressure_", "energy_")
#: Fields treated as signed, linear-range quantities (velocity).
VELOCITY_FIELDS = ("Uvelocity", "Wvelocity")

#: Default 8-channel set: the 6 material densities + the two velocity
#: components, matching
#: ``LSC_rho2rho_temporal_2frame_DataSet``'s default ``hydro_fields`` and the
#: ``train_ldrViT_2frame_8ch_lp.py`` harness.
DEFAULT_CHANNEL_LIST = [
    "density_case",
    "density_cushion",
    "density_maincharge",
    "density_outside_air",
    "density_striker",
    "density_throw",
    "Uvelocity",
    "Wvelocity",
]


def classify_channel(channel: str) -> str:
    """Classify a channel name into a normalization family.

    Args:
        channel (str): Channel/field name (e.g. ``"density_case"``).

    Returns:
        str: One of ``"density"``, ``"velocity"``, or ``"log_range"``.

    Raises:
        ValueError: If the channel does not match a known family.
    """
    if channel.startswith(DENSITY_PREFIX):
        return "density"
    if channel in VELOCITY_FIELDS:
        return "velocity"
    if channel.startswith(LOG_RANGE_PREFIXES):
        return "log_range"
    raise ValueError(
        f"Channel {channel!r} does not match a known normalization family "
        f"(density prefix {DENSITY_PREFIX!r}, velocity fields {VELOCITY_FIELDS}, "
        f"log-range prefixes {LOG_RANGE_PREFIXES})."
    )


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """Apply a signed log1p transform: ``sign(x) * log1p(|x|)``.

    Preserves ``x == 0 -> 0`` and the sign of ``x``.

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Transformed array.
    """
    return np.sign(x) * np.log1p(np.abs(x))


def compute_channel_stats(
    channel: str,
    voxels: np.ndarray,
    density_percentile: float,
    low_percentile: float,
    high_percentile: float,
) -> dict[str, float | str]:
    """Compute the normalization record for one channel's pooled voxel values.

    Args:
        channel (str): Channel/field name.
        voxels (np.ndarray): Flattened, pooled voxel values across all sampled
            frames for this channel.
        density_percentile (float): Percentile (0-100) of nonzero voxel
            magnitudes used as the one-sided density scale factor.
        low_percentile (float): Low percentile (0-100) used for the symmetric
            velocity/log-range scale factor.
        high_percentile (float): High percentile (0-100) used for the
            symmetric velocity/log-range scale factor.

    Returns:
        dict[str, float | str]: Normalization record with keys ``"family"``
        (one of ``"density"``, ``"velocity"``, ``"log_range"``) and
        ``"scale"`` (the positive scale factor; ``normalized = raw / scale``
        for density/velocity, or
        ``normalized = sign(raw) * log1p(|raw|) / scale`` for log-range
        fields).

    Raises:
        ValueError: If no finite voxel values are available to estimate the
            scale (e.g. a density channel with no nonzero voxels in the
            sample).
    """
    family = classify_channel(channel)
    voxels = voxels[np.isfinite(voxels)]

    if family == "density":
        nonzero = voxels[voxels != 0.0]
        if nonzero.size == 0:
            raise ValueError(
                f"No nonzero voxels sampled for density channel {channel!r}; "
                "cannot estimate a scale factor. Increase the sample count or "
                "check the channel name."
            )
        scale = float(np.percentile(np.abs(nonzero), density_percentile))
    elif family == "velocity":
        if voxels.size == 0:
            raise ValueError(f"No voxels sampled for channel {channel!r}.")
        p_low = np.percentile(voxels, low_percentile)
        p_high = np.percentile(voxels, high_percentile)
        scale = float(max(abs(p_low), abs(p_high)))
    else:  # "log_range"
        if voxels.size == 0:
            raise ValueError(f"No voxels sampled for channel {channel!r}.")
        transformed = signed_log1p(voxels)
        p_low = np.percentile(transformed, low_percentile)
        p_high = np.percentile(transformed, high_percentile)
        scale = float(max(abs(p_low), abs(p_high)))

    # Guard against a degenerate (all-zero/constant) sample.
    if scale <= 0.0:
        raise ValueError(
            f"Computed a non-positive scale ({scale}) for channel {channel!r}; "
            "the sampled voxels may be degenerate (all zero/constant)."
        )

    return {"family": family, "scale": scale}


def save_channel_norm(fileout: str, stats: dict[str, dict[str, float | str]]) -> None:
    """Save per-channel normalization records to an NPZ file.

    Each channel's record is saved under ``f"{channel}__family"`` /
    ``f"{channel}__scale"`` keys, plus a ``"channels"`` key listing the
    channel names in order.

    Args:
        fileout (str): Destination ``.npz`` path.
        stats (dict[str, dict[str, float | str]]): Mapping from channel name to
            its normalization record, as returned by
            :func:`compute_channel_stats`.
    """
    save_kwargs: dict[str, np.ndarray] = {"channels": np.array(list(stats.keys()))}
    for channel, record in stats.items():
        save_kwargs[f"{channel}__family"] = np.array(record["family"])
        save_kwargs[f"{channel}__scale"] = np.array(record["scale"], dtype=np.float64)

    np.savez(fileout, **save_kwargs)


def load_channel_norm(filepath: str) -> dict[str, dict[str, float | str]]:
    """Load per-channel normalization records saved by :func:`save_channel_norm`.

    Args:
        filepath (str): Path to the ``.npz`` file.

    Returns:
        dict[str, dict[str, float | str]]: Mapping from channel name to its
        normalization record (``{"family": str, "scale": float}``).
    """
    npz = np.load(filepath)
    channels = npz["channels"].tolist()
    stats = {}
    for channel in channels:
        stats[channel] = {
            "family": str(npz[f"{channel}__family"]),
            "scale": float(npz[f"{channel}__scale"]),
        }
    return stats


def normalize_field(raw: np.ndarray, record: dict[str, float | str]) -> np.ndarray:
    """Apply a saved normalization record to a raw field.

    Args:
        raw (np.ndarray): Raw (un-normalized) field values.
        record (dict[str, float | str]): Normalization record for this
            channel, as returned by :func:`compute_channel_stats` /
            :func:`load_channel_norm`.

    Returns:
        np.ndarray: Normalized field. Density/velocity fields are divided by
        the scale; log-range fields are signed-log1p transformed, then divided
        by the scale.
    """
    scale = record["scale"]
    if record["family"] == "log_range":
        return signed_log1p(raw) / scale
    return raw / scale
