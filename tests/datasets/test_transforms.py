"""Test data transforms."""

import torch

from yoke.datasets import transforms


def test_resize_pad_crop() -> None:
    """Ensure resize pad crop transform works as intended."""
    image_size = (1120, 400)
    scale_factor = 0.5
    scaled_image_size = (511, 230)  # arbitrary choice to test pad and crop
    tform = transforms.ResizePadCrop(
        interp_kwargs={"scale_factor": scale_factor}, scaled_image_size=scaled_image_size
    )
    image = torch.randn(
        (2, 3, 4, *image_size)
    )  # [batch, sequence length, variables, Y, X]
    out = tform(image)
    assert out.shape[-2:] == scaled_image_size, (
        "ResizePadCrop() output not the expected size!"
    )
