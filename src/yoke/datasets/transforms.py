"""Custom transforms for use in PyTorch Datasets."""

from collections.abc import Iterable

import torch


class ResizePadCrop(torch.nn.Module):
    """Resize image and pad/crop to desired final size."""

    def __init__(
        self,
        scale_factor: float = 1.0,
        scaled_image_size: Iterable[int, int] = None,
        pad_mode: str = "constant",
        pad_value: float = 0.0,
    ) -> None:
        """Initialize transform."""
        super().__init__()
        self.scale_factor = scale_factor
        self.scaled_image_size = scaled_image_size
        self.pad_mode = pad_mode
        self.pad_value = pad_value

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Resize, pad, and crop input `img` to desired size."""
        # Resize image.
        img = torch.nn.functional.interpolate(input=img, scale_factor=self.scale_factor)

        # Pad and crop image to desired size.
        if self.scaled_image_size is not None:
            # Pad:
            img = torch.nn.functional.pad(
                img,
                pad=(
                    0,
                    max(0, img.shape[1] - self.scaled_image_size[1]),
                    0,
                    max(0, img.shape[0] - self.scaled_image_size[0]),
                ),
                mode=self.pad_mode,
                value=self.pad_value,
            )

            # Crop:
            img = img[..., : self.scaled_image_size[0], : self.scaled_image_size[1]]

        return img
