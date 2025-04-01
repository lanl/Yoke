"""Custom transforms for use in PyTorch Datasets."""

from collections.abc import Iterable

import torch


class ResizePadCrop(torch.nn.Module):
    """Resize image and pad/crop to desired final size.

    This transform will resize an input image (same scale factor along each
    spatial dimension), pad the image, and then crop to the desired shape.

    Args:
        interp_kwargs (dict): Keyword arguments passed to
            torch.nn.functional.interpolate().
        scaled_image_size (Iterable[int, int]): Desired shape of the output image.
        pad_mode (str): Padding mode passed to torch.functional.nn.pad()
        pad_value (float): Padding value used by torch.functional.nn.pad() when
            pad_mode=="constant".
        pad_position (Iterable[str, str]): Pad value locations (i.e.,
            "top" or "bottom" for dim0, "left" or "right" for dim1).  The default
            ("bottom", "right") corresponds to the geometry of the LSC data where most
            dynamics occur in the bottom-left of the image.
    """

    def __init__(
        self,
        interp_kwargs: dict = {"scale_factor": 1.0},
        scaled_image_size: Iterable[int, int] = None,
        pad_mode: str = "constant",
        pad_value: float = 0.0,
        pad_position: Iterable[str, str] = ("bottom", "right"),
    ) -> None:
        """Initialize transform."""
        super().__init__()
        self.interp_kwargs = interp_kwargs
        self.scaled_image_size = scaled_image_size
        self.pad_mode = pad_mode
        self.pad_value = pad_value

        assert pad_position[0] in [
            "top",
            "bottom",
        ], "`pad_position[0]` must be either 'top' or 'bottom'!"
        assert pad_position[1] in [
            "left",
            "right",
        ], "`pad_position[1]` must be either 'left' or 'right'!"
        self.pad_position = pad_position

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Resize, pad, and crop input `img` to desired size."""
        # Resize image.
        img = torch.nn.functional.interpolate(input=img, **self.interp_kwargs)

        # Pad and crop image to desired size.
        if self.scaled_image_size is not None:
            # Pad (note the flipped dimension order convention used by pad()):
            pad_tb = max(0, self.scaled_image_size[0] - img.shape[-2])
            pad_lr = max(0, self.scaled_image_size[1] - img.shape[-1])
            img = torch.nn.functional.pad(
                img,
                pad=(
                    pad_lr if self.pad_position[1] == "left" else 0,
                    pad_lr if self.pad_position[1] == "right" else 0,
                    pad_tb if self.pad_position[0] == "top" else 0,
                    pad_tb if self.pad_position[0] == "bottom" else 0,
                ),
                mode=self.pad_mode,
                value=self.pad_value,
            )

            # Crop, ensuring we remove edges corresponding to the padding positions:
            if self.pad_position[0] == "left":
                img = img[..., -self.scaled_image_size[1] :]
            else:
                img = img[..., : self.scaled_image_size[1]]
            if self.pad_position[0] == "bottom":
                img = img[..., -self.scaled_image_size[0] :, :]
            else:
                img = img[..., : self.scaled_image_size[0], :]

        return img
