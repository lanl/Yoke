"""Training harness for MNIST digit classification."""

import argparse
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from yoke.models.mnist_model import mnist_CNN
from yoke.helpers import cli

if __name__ == "__main__":
    descr = "Train CNN on MNIST with flexible conv layers"
    parser = argparse.ArgumentParser(
        prog="MNIST Surrogate Training",
        description=descr,
        fromfile_prefix_chars="@",
    )
    # standard flags (gives you --studyIDX, --csv, --rundir, --cpFile)
    parser = cli.add_default_args(parser)
    # GPU/worker flags (e.g. --multigpu, --Ngpus, --num_workers)
    parser = cli.add_computing_args(parser)

    parser = cli.add_training_args(parser)

    # Model‐specific hyperparameters
    parser.add_argument("--conv1", type=int, default=32, help="1st conv channels")
    parser.add_argument("--conv2", type=int, default=64, help="2nd conv channels")
    parser.add_argument("--conv3", type=int, default=128, help="3rd conv channels")
    parser.add_argument("--conv4", type=int, default=128, help="4th conv channels")
    parser.add_argument("--lr", type=float, default=1.0, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.7, help="LR step decay")
    parser.add_argument(
        "--epochs", type=int, default=3, help="number of epochs to train"
    )
    parser.add_argument(
        "--data_dir", type=str, default="../data/MNIST", help="path to MNIST data"
    )
    parser.add_argument("--no_cuda", action="store_true", help="disable CUDA")
    parser.add_argument("--no_mps", action="store_true", help="disable macOS MPS")
    parser.add_argument("--dry_run", action="store_true", help="run only one batch")
    parser.add_argument(
        "--log_interval", type=int, default=10, help="batches between logs"
    )
    parser.add_argument(
        "--train_per_val",
        action="store",
        type=int,
        default=3,
        help="Number of training epochs between each validation epoch",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="save final model to f'mnist_studyXXX_epochYYY.pt'",
    )

    args = parser.parse_args()

    # Device selection
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")
    torch.manual_seed(0)

    # Data loaders
    train_kwargs = {"batch_size": args.batch_size, "shuffle": True}
    test_kwargs = {"batch_size": args.batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": args.num_workers, "pin_memory": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_ds = datasets.MNIST(
        args.data_dir, train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        args.data_dir, train=False, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(train_ds, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_ds, **test_kwargs)

    # Model, optimizer, scheduler
    model = mnist_CNN(
        conv1_size=args.conv1,
        conv2_size=args.conv2,
        conv3_size=args.conv3,
        conv4_size=args.conv4,
    ).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Optionally resume
    start_epoch = 1
    if args.continuation and args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        print("Resuming from", args.checkpoint)
        # (You could parse the epoch number out of the filename here.)


    # Initialize things to save
    trainbatch_ID = 0
    valbatch_ID = 0

    train_batchsize = train_loader.batch_size
    val_batchsize = test_loader.batch_size

    # mnist_study{args.studyIDX:03d}_epoch{args.epochs:03d}.pt"
    train_rcrd_filename = f"training_study{args.studyIDX:03d}_epoch{args.epochs:03d}.csv"
    #train_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")

    # Train on all training samples
    with open(train_rcrd_filename, "w") as train_rcrd_file:
        # Training loop
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                trainbatch_ID += 1
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

                # Stack loss record and write using numpy
                batch_records = np.column_stack(
                    [
                        epoch,  #np.full(train_batchsize, epoch),
                        trainbatch_ID,  #np.full(train_batchsize, trainbatch_ID),
                        loss.detach().cpu().numpy().flatten()
                    ]
                )

                np.savetxt(train_rcrd_file, batch_records, fmt="%d, %d, %.8f")

                if batch_idx % args.log_interval == 0:
                    print(
                        f"Train Epoch {epoch} [{batch_idx * len(data)}/"
                        f"{len(train_loader.dataset)}]\tLoss: {loss.item():.4f}"
                    )
                    if args.dry_run:
                        break

            # Evaluate on all validation samples
            if epoch % args.train_per_val == 0:
                print("Validating...", epoch)
                val_rcrd_filename = f"validation_study{args.studyIDX:03d}_epoch{args.epochs:03d}.csv"
                with open(val_rcrd_filename, "w") as val_rcrd_file:
                    with torch.no_grad():
                        # Validation loop
                        for batch_idx, (data, target) in enumerate(test_loader):
                            valbatch_ID += 1
                            data, target = data.to(device), target.to(device)
                            output = model(data)
                            loss = F.nll_loss(output, target)

                            # Stack loss record and write using numpy
                            batch_records = np.column_stack(
                                [
                                    epoch,  #np.full(train_batchsize, epoch),
                                    valbatch_ID,  #np.full(train_batchsize, trainbatch_ID),
                                    loss.detach().cpu().numpy().flatten()
                                ]
                            )
                            
                            np.savetxt(val_rcrd_file, batch_records, fmt="%d, %d, %.8f")

            # (You can add a test() call here if you like, mirroring the original.)

            scheduler.step()

    # Save
    if args.save_model:
        out_name = f"mnist_study{args.studyIDX:03d}_epoch{args.epochs:03d}.pt"
        torch.save(model.state_dict(), out_name)
        print("Saved model to", out_name)
