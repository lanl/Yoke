"""Helper tools for designing misc. training strategies."""

from collections.abc import Iterable

import numpy as np
import scipy.optimize


def validate_patch_and_window(
    image_size: Iterable[int, int],
    patch_size: Iterable[int, int],
    window_sizes: Iterable[
        Iterable[int, int],
        Iterable[int, int],
        Iterable[int, int],
        Iterable[int, int],
    ],
    patch_merge_scales: Iterable[
        Iterable[int, int],
        Iterable[int, int],
        Iterable[int, int],
    ] = None,
) -> np.array:
    """Validate LodeRunner design parameters with respect to image size.

    Ensures compatibility between :code:`patch_size` and :code:`window_sizes` in
    relation to the overall :code:`image_size` for the LodeRunner architecture.

    Specifically, this function verifies that:

    - The input image dimensions are divisible by the patch size
    - Each SWIN encoder/decoder stage's window size is compatible with the spatial
      resolution
    - Patch merging scales align correctly across stages

    .. note::
        All dimensions (image size, patch size, window size, and merge scale) must
        be chosen such that intermediate feature maps maintain integer dimensions
        after each stage.

    Args:
        image_size (Iterable[int, int]): Height and width (in pixels) of the input
            image, typically expressed as :math:`(H, W)`.

        patch_size (Iterable[int, int]): Patch height and width used in the initial
            embedding, expressed as :math:`(p_h, p_w)`. Must divide :math:`H` and
            :math:`W` evenly.

        window_sizes (List[Tuple[int, int]]): A list of four window sizes (height,
            width), one for each SWIN encoder/decoder stage. Each tuple should
            represent :math:`(w_h, w_w)` and be compatible with the feature map
            resolution at that stage.

        patch_merge_scales (List[Tuple[int, int]]): A list of three scaling factors
            used in patch merging, one per merge layer. Each tuple defines height
            and width downscaling, i.e., :math:`(s_h, s_w)`.

    """
    # Walk through LodeRunner stages and verify size compatibility.
    valid = np.zeros((len(window_sizes), 2, 2), dtype=bool)  # [stage, action, dimension]
    image_size = np.array(image_size)
    patch_size = np.array(patch_size)
    window_sizes = np.array(window_sizes)
    patch_merge_scales = (
        np.array(patch_merge_scales)
        if patch_merge_scales
        else np.array([(2, 2) for _ in range(len(window_sizes) - 1)])
    )

    # Stage 1 patching (dividing input image into patches):
    valid[0, 0] = (image_size % patch_size) == 0
    new_grid = image_size // patch_size

    # Stage 1 windowing (grouping patches into a window view):
    valid[0, 1] = (new_grid % window_sizes[0, :]) == 0

    # Stages 2-4 patch merging and windowing:
    for s in range(1, 4):
        # Patch merging (grouping patches into new, larger patches):
        valid[s, 0] = (new_grid % patch_merge_scales[s - 1, :]) == 0
        new_grid = new_grid // patch_merge_scales[s - 1, :]

        # Windowing:
        valid[s, 1] = (new_grid % window_sizes[s, :]) == 0

    return valid


def find_valid_pad(
    image_size: tuple[int, int],
    patch_size: tuple[int, int],
    window_sizes: Iterable,
    patch_merge_scales: Iterable,
    pad_options: Iterable,
) -> tuple[list, list]:
    """Search for pad values that make sizes of image, patch, and window compatible."""
    pad_dim0 = []
    for pad in pad_options:
        if np.all(
            validate_patch_and_window(
                image_size=image_size + np.array([pad, 0]),
                patch_size=patch_size,
                window_sizes=window_sizes,
                patch_merge_scales=patch_merge_scales,
            )[:, :, 0]
        ):
            pad_dim0.append(pad)

    pad_dim1 = []
    for pad in pad_options:
        if np.all(
            validate_patch_and_window(
                image_size=image_size + np.array([0, pad]),
                patch_size=patch_size,
                window_sizes=window_sizes,
                patch_merge_scales=patch_merge_scales,
            )[:, :, 1]
        ):
            pad_dim1.append(pad)

    return pad_dim0, pad_dim1


def choose_downsample_factor(
    image_size: np.array = np.array([1120, 400]),
    patch_size: int = 5,
    window_sizes: Iterable = [(2, 2) for _ in range(4)],
    patch_merge_scales: Iterable = [(2, 2) for _ in range(3)],
    pad_options: np.array = np.arange(100),
    desired_scale_factor: float = 1.0,
    max_scale_dev: float = 0.5,
) -> float:
    """Choose downsample factor close to desired factor to minimize padding."""

    # Define a loss function for optimizing scale factor.
    def loss(scale_factor: float) -> int:
        ds_size = np.floor(image_size * scale_factor)
        pad_dim0, pad_dim1 = find_valid_pad(
            image_size=ds_size,
            patch_size=patch_size,
            window_sizes=window_sizes,
            patch_merge_scales=patch_merge_scales,
            pad_options=pad_options,
        )

        # Define the loss as the total number of padding pixels.
        return (
            pad_dim0[0] * ds_size[1]
            + pad_dim1[0] * ds_size[0]
            + pad_dim0[0] * pad_dim1[0]
        )

    # Search for a scale factor close to the requested scale factor.
    res = scipy.optimize.minimize(
        loss,
        x0=[desired_scale_factor],
        bounds=[
            (
                desired_scale_factor * (1.0 - max_scale_dev),
                desired_scale_factor * (1.0 + max_scale_dev),
            )
        ],
        method="Nelder-Mead",
    )
    new_scale = res.x.item()

    # Compute resulting scaled image size.
    rescaled_size = np.floor(new_scale * image_size).astype(int)
    pad_dim0, pad_dim1 = find_valid_pad(
        image_size=rescaled_size,
        patch_size=patch_size,
        window_sizes=window_sizes,
        patch_merge_scales=patch_merge_scales,
        pad_options=pad_options,
    )
    scaled_image_size = rescaled_size + np.array([pad_dim0[0], pad_dim1[0]])

    return new_scale, scaled_image_size


if __name__ == "__main__":
    # Search for a scale factor close to 0.25 that minimizes padding needed
    # for LodeRunner.
    scale_factor = 0.25
    max_scale_dev = 0.2
    image_size = np.array([1120, 400])
    patch_size = 5
    window_sizes = [(2, 2) for _ in range(4)]
    patch_merge_scales = [(2, 2) for _ in range(3)]
    pad_options = np.arange(stop=np.max(image_size) // 8)
    new_scale, scaled_image_size = choose_downsample_factor(
        image_size=image_size,
        patch_size=patch_size,
        window_sizes=window_sizes,
        patch_merge_scales=patch_merge_scales,
        pad_options=pad_options,
        desired_scale_factor=scale_factor,
        max_scale_dev=max_scale_dev,
    )
    print(f"Suggested scale factor = {new_scale:.6f}")
    print(f"Rescaled image size = {scaled_image_size}")
