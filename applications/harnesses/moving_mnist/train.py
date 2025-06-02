# python applications/harnesses/moving_mnist/train.py
# python applications/evaluation/TandVplot.py -S --basedir . -I 0 -Y 2 -Nt 2 -Nv 250
import torch
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from torchvision.datasets import MovingMNIST
from torch.utils.data import Dataset

from yoke.models.vit.swin.bomberman import LodeRunner
import yoke.torch_training_utils as tr
import yoke.helpers.logger as yl


class mmnist_dataSet(Dataset):
    """Moving MNIST dataset."""

    def __init__(self, fraction=1, fraction_side="left") -> None:
        self.mmnist = MovingMNIST(".", download=True)
        total_len = 1000
        seq_len = 20
        pairs_per_seq = seq_len - 1
        frac_range = range(0, int(fraction * total_len))
        if fraction_side == "right":
            frac_range = range(int(fraction * total_len), total_len)
        self.seq_id = [
            x for xs in [np.repeat(i, pairs_per_seq) for i in frac_range] for x in xs
        ]
        pairs_local = [
            sliding_window_view(np.arange(0, seq_len), window_shape=2)
            for _ in frac_range
        ]
        self.pairs_local = np.concatenate(pairs_local)

    def __len__(self) -> int:
        """Return effectively infinite number of samples in dataset."""
        return len(self.seq_id)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a tuple of a batch's input and output data."""
        img_pair = self.mmnist[self.seq_id[index]][self.pairs_local[index]]
        start_img = (
            torch.tensor(np.expand_dims(img_pair[0, 0, ...], 0)).to(torch.float32) / 255
        )
        end_img = (
            torch.tensor(np.expand_dims(img_pair[1, 0, ...], 0)).to(torch.float32) / 255
        )
        Dt = torch.tensor(0.25, dtype=torch.float32)  # arbitrary value
        return start_img, end_img, Dt


# breakpoint()
# (a, b, c) = next(iter(mmnist_dataSet()))

if __name__ == "__main__":
    yl.configure_logger("yoke_logger", level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LodeRunner(
        default_vars=["var1"],
        image_size=(64, 64),
        patch_size=(8, 8),
        embed_dim=4,
        emb_factor=2,
        num_heads=2,
        block_structure=(1, 1, 3, 1),
        window_sizes=[
            (4, 4),
            (4, 4),
            (2, 2),
            (1, 1),
        ],
        patch_merge_scales=[
            (2, 2),
            (2, 2),
            (2, 2),
        ],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01,
    )

    # Use `reduction='none'` so loss on each sample in batch can be recorded.
    loss_fn = nn.MSELoss(reduction="none")

    model.to(device)

    # initialize outside of epoch loop because this is a single channel only
    train_dataset = mmnist_dataSet(0.75, "left")
    val_dataset = mmnist_dataSet(0.25, "right")

    train_dataloader = tr.make_dataloader(
        dataset=train_dataset,
        batch_size=2,
        num_batches=250,
        num_workers=1,
        prefetch_factor=2,
    )
    val_dataloader = tr.make_dataloader(
        dataset=val_dataset,
        batch_size=2,
        num_batches=25,
        num_workers=1,
        prefetch_factor=2,
    )

    num_epochs = 100
    channel_map = [0]
    for epochIDX in tqdm(range(num_epochs)):
        tr.train_simple_loderunner_epoch(
            channel_map=channel_map,
            training_data=train_dataloader,
            validation_data=val_dataloader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochIDX=epochIDX,
            train_per_val=10,
            train_rcrd_filename="train.csv",
            val_rcrd_filename="val.csv",
            device=device,
            verbose=False,
        )
        torch.cuda.empty_cache()

    (start_img, true_img, Dt) = next(iter(train_dataloader))
    channel_map = [0]
    pred_img = model(
        start_img.to(device),
        torch.tensor(channel_map).to(device, non_blocking=True),
        torch.tensor(channel_map).to(device, non_blocking=True),
        Dt.to(device),
    )

    plt.imshow(pred_img[1,0,...].detach().cpu())
    plt.savefig("test.pdf")
