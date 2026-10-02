"""Create diagnostic frames and an animation from a chained LodeRunner rollout."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as mpl_animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch

from yoke.datasets.lsc_dataset import LSCread_npz_NaN, volfrac_density
from yoke.models.vit.swin.bomberman import LodeRunner, LodeRunnerViT
from yoke.utils.checkpointing import load_model_and_optimizer


TIMESTEP_DELTA = 0.25
AGGREGATE_FIELDS = {"sum_density", "sum_energy", "sum_pressure"}


def build_parser() -> argparse.ArgumentParser:
    """Build the chained-rollout command-line parser."""
    parser = argparse.ArgumentParser(
        prog="LodeRunner chained animation",
        description=(
            "Run a chained LodeRunner prediction and create per-channel PNGs "
            "plus an MP4 or GIF animation."
        ),
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint created by yoke.utils.checkpointing.save_model_and_optimizer.",
    )
    parser.add_argument(
        "--LSC_NPZ_DIR",
        type=Path,
        required=True,
        help="Directory containing the LSC NPZ files.",
    )
    parser.add_argument(
        "--simulation_prefix",
        required=True,
        help="Simulation prefix, for example lsc240420_id00201.",
    )
    parser.add_argument(
        "--prediction_length",
        type=int,
        required=True,
        help="Number of new autoregressive frames to predict.",
    )
    parser.add_argument(
        "--num_input_frames",
        type=int,
        choices=(1, 2),
        required=True,
        help="Number of input frames expected by the checkpoint model.",
    )
    parser.add_argument(
        "--output_filename",
        type=Path,
        required=True,
        help="Output animation filename ending in .mp4 or .gif.",
    )
    parser.add_argument(
        "--movie_field",
        default="sum_density",
        help=(
            "Exact model channel to animate, or one of sum_density, sum_energy, "
            "and sum_pressure."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=4.0,
        help="Animation frames per second (default: 4).",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Validate arguments that argparse cannot constrain."""
    if args.prediction_length < 1:
        raise ValueError("prediction_length must be at least 1.")
    if args.fps <= 0:
        raise ValueError("fps must be greater than 0.")
    if args.output_filename.suffix.lower() not in {".mp4", ".gif"}:
        raise ValueError("output_filename must end in .mp4 or .gif.")
    if not args.LSC_NPZ_DIR.is_dir():
        raise FileNotFoundError(f"LSC NPZ directory not found: {args.LSC_NPZ_DIR}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")


def _load_model(
    checkpoint: Path, device: torch.device, num_input_frames: int
) -> tuple[torch.nn.Module, int, list[str]]:
    """Load a current Yoke checkpoint and validate its rollout metadata."""
    model, _, checkpoint_epoch = load_model_and_optimizer(
        str(checkpoint),
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-6},
        available_models={
            "LodeRunner": LodeRunner,
            "LodeRunnerViT": LodeRunnerViT,
        },
        device=device,
    )

    fields = list(getattr(model, "default_vars", []))
    if not fields:
        raise ValueError(
            "Checkpoint model metadata must provide non-empty default_vars."
        )

    model_input_frames = int(getattr(model, "num_input_frames", 1))
    if model_input_frames != num_input_frames:
        raise ValueError(
            "num_input_frames does not match the checkpoint model: "
            f"input requested {num_input_frames}, model expects {model_input_frames}."
        )

    model.eval()
    return model, checkpoint_epoch, fields


def _npz_path(npz_dir: Path, simulation_prefix: str, time_index: int) -> Path:
    """Return the expected path for one LSC simulation frame."""
    return npz_dir / f"{simulation_prefix}_pvi_idx{time_index:05d}.npz"


def _load_frame(
    npz_path: Path, fields: list[str]
) -> tuple[torch.Tensor, float, np.ndarray, np.ndarray]:
    """Load one frame using the same preprocessing as the LSC temporal datasets."""
    if not npz_path.is_file():
        raise FileNotFoundError(f"Required LSC frame not found: {npz_path}")

    with np.load(npz_path) as npz:
        field_images = []
        for field in fields:
            image = LSCread_npz_NaN(npz, field)
            field_images.append(volfrac_density(image, npz, field))

        sim_time = float(np.asarray(npz["sim_time"]).squeeze())
        r_coord = np.asarray(npz["Rcoord"])
        z_coord = np.asarray(npz["Zcoord"])

    frame = torch.tensor(np.stack(field_images, axis=0), dtype=torch.float32)
    return frame, sim_time, r_coord, z_coord


def _field_indices(movie_field: str, fields: list[str]) -> tuple[list[int], str]:
    """Resolve an exact or aggregate movie field to model channel indices."""
    if movie_field in fields:
        return [fields.index(movie_field)], movie_field

    if movie_field not in AGGREGATE_FIELDS:
        valid_fields = ", ".join([*fields, *sorted(AGGREGATE_FIELDS)])
        raise ValueError(
            f"Unknown movie_field {movie_field!r}. Valid values: {valid_fields}"
        )

    field_term = movie_field.removeprefix("sum_")
    indices = [
        index for index, field in enumerate(fields) if field_term in field.lower()
    ]
    if not indices:
        raise ValueError(
            f"movie_field {movie_field!r} has no matching checkpoint channels."
        )
    return indices, movie_field


def _select_field(frame: torch.Tensor, indices: list[int]) -> np.ndarray:
    """Select one channel or sum a group of channels for plotting."""
    return frame[indices].sum(dim=0).detach().cpu().numpy()


def _safe_name(field: str) -> str:
    """Convert a channel name into a safe output-directory component."""
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in field
    )
    return safe or "channel"


def _render_panel(
    truth: np.ndarray | None,
    prediction: np.ndarray,
    sim_time: float,
    r_coord: np.ndarray,
    z_coord: np.ndarray,
    field_label: str,
    output_path: Path,
) -> None:
    """Render a comparison panel, leaving unavailable truth panels blank."""
    if truth is None:
        discrepancy = None
        value_min = float(prediction.min())
        value_max = float(prediction.max())
    else:
        discrepancy = np.abs(truth - prediction)
        value_min = float(min(truth.min(), prediction.min()))
        value_max = float(max(truth.max(), prediction.max()))
    if value_min == value_max:
        value_max = value_min + 1.0

    extent = [
        float(r_coord.min()),
        float(r_coord.max()),
        float(z_coord.min()),
        float(z_coord.max()),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    figure.suptitle(f"{field_label}: T={sim_time:.2f} us", fontsize=18)

    axes[0].set_title("Truth", fontsize=16)
    axes[0].set_xlabel("R-axis (cm)")
    axes[0].set_ylabel("Z-axis (cm)")

    prediction_image = axes[1].imshow(
        prediction,
        aspect="equal",
        extent=extent,
        origin="lower",
        cmap="viridis",
        vmin=value_min,
        vmax=value_max,
    )
    axes[1].set_title("Prediction", fontsize=16)
    axes[1].set_xlabel("R-axis (cm)")
    axes[1].tick_params(axis="y", left=False, labelleft=False)

    axes[2].set_title("Absolute discrepancy", fontsize=16)
    axes[2].set_xlabel("R-axis (cm)")
    axes[2].tick_params(axis="y", left=False, labelleft=False)

    if truth is None:
        for axes_index in (0, 2):
            axes[axes_index].set_facecolor("white")
            axes[axes_index].set_xlim(extent[0], extent[1])
            axes[axes_index].set_ylim(extent[2], extent[3])
            axes[axes_index].text(
                0.5,
                0.5,
                "Ground truth unavailable",
                ha="center",
                va="center",
                transform=axes[axes_index].transAxes,
            )
        figure.colorbar(prediction_image, ax=axes[1], location="bottom", shrink=0.75)
    else:
        truth_image = axes[0].imshow(
            truth,
            aspect="equal",
            extent=extent,
            origin="lower",
            cmap="viridis",
            vmin=value_min,
            vmax=value_max,
        )
        figure.colorbar(truth_image, ax=axes[:2], location="bottom", shrink=0.75)

        discrepancy_max = float(discrepancy.max())
        if discrepancy_max == 0.0:
            discrepancy_max = 1.0
        discrepancy_image = axes[2].imshow(
            discrepancy,
            aspect="equal",
            extent=extent,
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=discrepancy_max,
        )
        figure.colorbar(discrepancy_image, ax=axes[2], location="bottom", shrink=0.75)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _predict_next(
    model: torch.nn.Module,
    history: list[torch.Tensor],
    num_input_frames: int,
    channel_indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Predict the next frame from the current autoregressive history."""
    if num_input_frames == 1:
        model_input = history[-1].unsqueeze(0)
        lead_times = torch.tensor([TIMESTEP_DELTA], device=device)
    else:
        model_input = torch.stack(history[-2:], dim=0).unsqueeze(0)
        lead_times = torch.tensor([[TIMESTEP_DELTA, TIMESTEP_DELTA]], device=device)

    prediction = model(model_input, channel_indices, channel_indices, lead_times)
    return prediction.squeeze(0)


def _write_animation(frame_paths: list[Path], output_path: Path, fps: float) -> None:
    """Assemble rendered movie panels into an MP4 or GIF."""
    suffix = output_path.suffix.lower()
    writer_name = "ffmpeg" if suffix == ".mp4" else "imagemagick"
    if not mpl_animation.writers.is_available(writer_name):
        raise RuntimeError(
            f"Matplotlib writer {writer_name!r} is unavailable; cannot create "
            f"{suffix} output. Install the corresponding FFmpeg or ImageMagick "
            "executable."
        )

    first_frame = mpimg.imread(frame_paths[0])
    height, width = first_frame.shape[:2]
    dpi = 100
    figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    image = axes.imshow(first_frame)
    axes.axis("off")

    writer = mpl_animation.writers[writer_name](fps=fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with writer.saving(figure, str(output_path), dpi=dpi):
            for frame_path in frame_paths:
                image.set_data(mpimg.imread(frame_path))
                writer.grab_frame()
    finally:
        plt.close(figure)


def run(args: argparse.Namespace) -> None:
    """Run chained inference and write all requested visualizations."""
    _validate_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_epoch, fields = _load_model(
        args.checkpoint, device, args.num_input_frames
    )
    movie_indices, movie_label = _field_indices(args.movie_field, fields)

    frames_root = args.output_filename.parent / f"{args.output_filename.stem}_frames"
    history = []
    for time_index in range(args.num_input_frames):
        seed, _, r_coord, z_coord = _load_frame(
            _npz_path(args.LSC_NPZ_DIR, args.simulation_prefix, time_index), fields
        )
        history.append(seed.to(device))

    channel_indices = torch.arange(len(fields), dtype=torch.long, device=device)
    movie_paths = []
    print(
        f"Loaded checkpoint epoch {checkpoint_epoch} on {device}; "
        f"rolling out {args.prediction_length} frames."
    )

    with torch.inference_mode():
        for rollout_index in range(args.prediction_length):
            time_index = args.num_input_frames + rollout_index
            prediction = _predict_next(
                model,
                history,
                args.num_input_frames,
                channel_indices,
                device,
            )

            truth_path = _npz_path(args.LSC_NPZ_DIR, args.simulation_prefix, time_index)
            if truth_path.is_file():
                truth, sim_time, r_coord, z_coord = _load_frame(truth_path, fields)
                truth_cpu = truth.cpu()
            else:
                truth_cpu = None
                sim_time = time_index * TIMESTEP_DELTA
                print(
                    f"Ground truth unavailable for time index {time_index:05d}; "
                    "rendering blank truth and discrepancy panels."
                )

            prediction_cpu = prediction.cpu()
            for channel_index, channel_name in enumerate(fields):
                output_path = (
                    frames_root
                    / _safe_name(channel_name)
                    / f"frame_{time_index:05d}.png"
                )
                _render_panel(
                    (None if truth_cpu is None else truth_cpu[channel_index].numpy()),
                    prediction_cpu[channel_index].numpy(),
                    sim_time,
                    r_coord,
                    z_coord,
                    channel_name,
                    output_path,
                )

            movie_path = frames_root / "movie" / f"frame_{time_index:05d}.png"
            _render_panel(
                (None if truth_cpu is None else _select_field(truth_cpu, movie_indices)),
                _select_field(prediction_cpu, movie_indices),
                sim_time,
                r_coord,
                z_coord,
                movie_label,
                movie_path,
            )
            movie_paths.append(movie_path)

            history.append(prediction)
            history = history[-args.num_input_frames :]
            print(
                f"Rendered prediction {rollout_index + 1}/{args.prediction_length}: "
                f"time index {time_index:05d}"
            )

    _write_animation(movie_paths, args.output_filename, args.fps)
    print(f"Saved animation: {args.output_filename}")
    print(f"Saved diagnostic frames under: {frames_root}")


def main() -> None:
    """Parse command-line arguments and run the chained rollout."""
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
