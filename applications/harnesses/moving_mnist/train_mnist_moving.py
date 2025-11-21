# python applications/harnesses/moving_mnist/train.py
# python applications/harnesses/moving_mnist/train.py --continuation
# python applications/evaluation/TandVplot.py -S --basedir . -I 0 -Y 2 -Nt 2 -Nv 250
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from torchvision.datasets import MovingMNIST
from torch.utils.data import Dataset
from collections.abc import Iterator

from yoke.helpers import cli
import yoke.helpers.logger as yl
import yoke.utils.dataload as dl
import yoke.utils.checkpointing as ch
from yoke.models.vit.swin.bomberman import LodeRunner
from yoke.utils.training.epoch.loderunner import train_simple_loderunner_epoch


class mmnist_dataSet(Dataset):
    """Moving MNIST dataset."""

    def __init__(self, fraction: float = 1, fraction_side: str = "left") -> None:
        """Setup the data."""
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # standard flags (gives you --studyIDX, --csv, --rundir, --cpFile)
    parser = cli.add_default_args(parser)
    # GPU/worker flags (e.g. --multigpu, --Ngpus, --num_workers)
    parser = cli.add_computing_args(parser)
    parser = cli.add_training_args(parser)

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--epochs", type=int, default=20, help="number of epochs to train"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../../data/MovingMNIST",
        help="path to MNIST data",
    )

    args = parser.parse_args()
    checkpoint = args.checkpoint
    CONTINUATION = args.continuation

    yl.configure_logger("yoke_logger", level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_args = {
        "default_vars": ["var1"],
        "image_size": (64, 64),
        "patch_size": (8, 8),
        "embed_dim": 4,
        "emb_factor": 2,
        "num_heads": 2,
        "block_structure": (1, 1, 3, 1),
        "window_sizes": [
            (4, 4),
            (4, 4),
            (2, 2),
            (1, 1),
        ],
        "patch_merge_scales": [
            (2, 2),
            (2, 2),
            (2, 2),
        ],
    }
    model = LodeRunner(**model_args)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01,
    )

    # Use `reduction='none'` so loss on each sample in batch can be recorded.
    loss_fn = nn.MSELoss(reduction="none")

    if CONTINUATION:
        available_models = {"LodeRunner": LodeRunner}
        model, starting_epoch = ch.load_model_and_optimizer(
            checkpoint,
            optimizer,
            available_models,
            device=device,
        )
        print("Model state loaded for continuation.")
    else:
        starting_epoch = 0
        model.to(device)

    # initialize outside of epoch loop because this is a single channel only
    train_dataset = mmnist_dataSet(0.75, "left")
    val_dataset = mmnist_dataSet(0.25, "right")

    train_dataloader = dl.make_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_batches=250,
        num_workers=1,
        prefetch_factor=2,
    )
    val_dataloader = dl.make_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_batches=25,
        num_workers=1,
        prefetch_factor=2,
    )

    channel_map = [0]
    for epoch_idx in tqdm(range(starting_epoch, starting_epoch + args.epochs)):
        train_simple_loderunner_epoch(
            channel_map=channel_map,
            training_data=train_dataloader,
            validation_data=val_dataloader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochIDX=epoch_idx,
            train_per_val=10,
            train_rcrd_filename="train.csv",
            val_rcrd_filename="val.csv",
            device=device,
            verbose=False,
        )
        torch.cuda.empty_cache()

        if epoch_idx % 10 == 0:

            def last_row(path_file: str = "train.csv") -> Iterator[str]:
                """Get last row of file."""
                with open(path_file) as f:
                    for line in f:
                        pass
                    last_line = line
                    yield last_line

            train_loss = np.loadtxt(last_row(), delimiter=",")[-1]
            # pred_loss =
            (start_img, true_img, Dt) = next(iter(train_dataloader))
            channel_map = [0]
            pred_img = model(
                start_img.to(device),
                torch.tensor(channel_map).to(device, non_blocking=True),
                torch.tensor(channel_map).to(device, non_blocking=True),
                Dt.to(device),
            )
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(start_img[1, 0, ...].detach().cpu())
            ax2.imshow(true_img[1, 0, ...].detach().cpu())
            ax3.imshow(pred_img[1, 0, ...].detach().cpu())
            plt.savefig(
                "pred-img_epoch-{}_train-loss-{}_val-loss-{}.pdf".format(
                    str(epoch_idx), str(train_loss), "placeholder"
                )
            )

    model.to("cpu")
    # Move optimizer state back to CPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to("cpu")

    # Save model and optimizer state
    ch.save_model_and_optimizer(
        model,
        optimizer,
        epoch_idx,
        checkpoint,
        model_class=LodeRunner,
        model_args=model_args,
    )
