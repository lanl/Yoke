"""Functions to train and evaluate DiffusionLodeRunner over a single epoch."""

import torch
import numpy as np
import time
from contextlib import nullcontext

from yoke.utils.training.datastep.diff_loderunner import (
    train_diffusion_loderunner_datastep,
    eval_diffusion_loderunner_datastep,
    train_DDP_diffusion_loderunner_datastep,
    eval_DDP_diffusion_loderunner_datastep,
)


def train_simple_diffusion_loderunner_epoch(
    training_data: torch.utils.data.DataLoader,
    validation_data: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochIDX: int,
    train_per_val: int,
    train_rcrd_filename: str,
    val_rcrd_filename: str,
    device: torch.device,
    in_vars: torch.Tensor,
    out_vars: torch.Tensor,
    verbose: bool = False,
) -> None:
    """Training and validation epochs on the DiffusionLodeRunner architecture.

    Training and validation information is saved to successive CSV files.

    Args:
        training_data (torch.utils.data.DataLoader): training dataloader
        validation_data (torch.utils.data.DataLoader): validation dataloader
        model (torch.nn.Module): DiffusionLodeRunner model to train
        optimizer (torch.optim.Optimizer): optimizer for training set
        loss_fn (torch.nn.Module): loss function for training set (typically MSE)
        epochIDX (int): Index of current training epoch
        train_per_val (int): Number of Training epochs between each validation
        train_rcrd_filename (str): Name of CSV file to save training sample stats to
        val_rcrd_filename (str): Name of CSV file to save validation sample stats to
        device (torch.device): device index to select
        in_vars (torch.Tensor): Input variable indices for conditioning
        out_vars (torch.Tensor): Output variable indices for prediction
        verbose (bool): Flag to print diagnostic output.
    """
    # Initialize things to save
    trainbatch_ID = 0
    valbatch_ID = 0

    train_batchsize = training_data.batch_size
    val_batchsize = validation_data.batch_size

    train_rcrd_filename = train_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")
    # Train on all training samples
    with open(train_rcrd_filename, "a") as train_rcrd_file:
        for traindata in training_data:
            trainbatch_ID += 1

            # Time each batch and print to stdout
            if verbose:
                startTime = time.time()

            noise_gt, noise_pred, train_loss = train_diffusion_loderunner_datastep(
                traindata, model, optimizer, loss_fn, device, in_vars, out_vars
            )

            if verbose:
                endTime = time.time()
                batch_time = endTime - startTime
                print(
                    f"Batch {trainbatch_ID} time (seconds): {batch_time:.5f}", flush=True
                )

            if verbose:
                startTime = time.time()

            # Stack loss record and write using numpy
            batch_records = np.column_stack(
                [
                    np.full(train_batchsize, epochIDX),
                    np.full(train_batchsize, trainbatch_ID),
                    train_loss.detach().cpu().numpy().flatten(),
                ]
            )

            np.savetxt(train_rcrd_file, batch_records, fmt="%d, %d, %.8f")

            if verbose:
                endTime = time.time()
                record_time = endTime - startTime
                print(
                    f"Batch {trainbatch_ID} record time: {record_time:.5f}", flush=True
                )

    # Evaluate on all validation samples
    if epochIDX % train_per_val == 0:
        print("Validating...", epochIDX)
        val_rcrd_filename = val_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")
        with open(val_rcrd_filename, "a") as val_rcrd_file:
            with torch.no_grad():
                for valdata in validation_data:
                    valbatch_ID += 1
                    noise_gt, noise_pred, val_loss = eval_diffusion_loderunner_datastep(
                        valdata, model, loss_fn, device, in_vars, out_vars
                    )

                    # Stack loss record and write using numpy
                    batch_records = np.column_stack(
                        [
                            np.full(val_batchsize, epochIDX),
                            np.full(val_batchsize, valbatch_ID),
                            val_loss.detach().cpu().numpy().flatten(),
                        ]
                    )

                    np.savetxt(val_rcrd_file, batch_records, fmt="%d, %d, %.8f")


def train_DDP_diffusion_loderunner_epoch(
    training_data: torch.utils.data.DataLoader,
    validation_data: torch.utils.data.DataLoader,
    num_train_batches: int,
    num_val_batches: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    LRsched: torch.optim.lr_scheduler._LRScheduler,
    epochIDX: int,
    train_per_val: int,
    train_rcrd_filename: str,
    val_rcrd_filename: str,
    device: torch.device,
    rank: int,
    world_size: int,
    in_vars: torch.Tensor,
    out_vars: torch.Tensor,
) -> None:
    """Distributed data-parallel DiffusionLodeRunner Epoch.

    Function to complete a training epoch on the DiffusionLodeRunner architecture.
    Training and validation information is saved to successive CSV files.

    Args:
        training_data (torch.utils.data.DataLoader): training dataloader
        validation_data (torch.utils.data.DataLoader): validation dataloader
        num_train_batches (int): Number of batches in training epoch
        num_val_batches (int): Number of batches in validation epoch
        model (torch.nn.Module): DiffusionLodeRunner model to train
        optimizer (torch.optim.Optimizer): optimizer for training set
        loss_fn (torch.nn.Module): loss function for training set
        LRsched (torch.optim.lr_scheduler._LRScheduler): Learning-rate scheduler called
                                                         every training step.
        epochIDX (int): Index of current training epoch
        train_per_val (int): Number of Training epochs between each validation
        train_rcrd_filename (str): Name of CSV file to save training sample stats to
        val_rcrd_filename (str): Name of CSV file to save validation sample stats to
        device (torch.device): device index to select
        rank (int): rank of process
        world_size (int): number of total processes
        in_vars (torch.Tensor): Input variable indices for conditioning
        out_vars (torch.Tensor): Output variable indices for prediction
    """
    # Initialize things to save
    trainbatch_ID = 0
    valbatch_ID = 0

    # Training loop
    model.train()
    train_rcrd_filename = train_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")
    with (
        open(train_rcrd_filename, "a") if rank == 0 else nullcontext()
    ) as train_rcrd_file:
        for trainbatch_ID, traindata in enumerate(training_data):
            # Stop when number of training batches is reached
            if trainbatch_ID >= num_train_batches:
                break

            # Training
            noise_gt, noise_pred, train_losses = train_DDP_diffusion_loderunner_datastep(
                traindata,
                model,
                optimizer,
                loss_fn,
                device,
                rank,
                world_size,
                in_vars,
                out_vars,
            )

            # Increment the learning-rate scheduler
            LRsched.step()

            # Save training record (rank 0 only)
            if rank == 0:
                batch_records = np.column_stack(
                    [
                        np.full(len(train_losses), epochIDX),
                        np.full(len(train_losses), trainbatch_ID),
                        train_losses.cpu().numpy().flatten(),
                    ]
                )
                np.savetxt(train_rcrd_file, batch_records, fmt="%d, %d, %.8f")

    # Validation loop
    if epochIDX % train_per_val == 0:
        print("Validating...", epochIDX)
        val_rcrd_filename = val_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")
        model.eval()
        with (
            open(val_rcrd_filename, "a") if rank == 0 else nullcontext()
        ) as val_rcrd_file:
            with torch.no_grad():
                for valbatch_ID, valdata in enumerate(validation_data):
                    # Stop when number of validation batches is reached
                    if valbatch_ID >= num_val_batches:
                        break

                    noise_gt, noise_pred, val_losses = (
                        eval_DDP_diffusion_loderunner_datastep(
                            valdata,
                            model,
                            loss_fn,
                            device,
                            rank,
                            world_size,
                            in_vars,
                            out_vars,
                        )
                    )

                    # Save validation record (rank 0 only)
                    if rank == 0:
                        batch_records = np.column_stack(
                            [
                                np.full(len(val_losses), epochIDX),
                                np.full(len(val_losses), valbatch_ID),
                                val_losses.cpu().numpy().flatten(),
                            ]
                        )
                        np.savetxt(val_rcrd_file, batch_records, fmt="%d, %d, %.8f")


def eval_diffusion_loderunner_epoch(
    testing_data: torch.utils.data.DataLoader,
    num_test_batches: int,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    epochIDX: int,
    test_rcrd_filename: str,
    device: torch.device,
    in_vars: torch.Tensor,
    out_vars: torch.Tensor,
) -> None:
    """DiffusionLodeRunner Evaluation-Only Epoch.

    Function to complete a testing epoch on the DiffusionLodeRunner architecture.
    Testing information is saved to successive CSV files.

    Args:
        testing_data (torch.utils.data.DataLoader): testing dataloader
        num_test_batches (int): Number of batches in testing epoch
        model (torch.nn.Module): DiffusionLodeRunner model to evaluate
        loss_fn (torch.nn.Module): loss function for testing set
        epochIDX (int): Index of current testing epoch
        test_rcrd_filename (str): Name of CSV file to save testing sample stats to
        device (torch.device): device index to select
        in_vars (torch.Tensor): Input variable indices for conditioning
        out_vars (torch.Tensor): Output variable indices for prediction
    """
    # Initialize things to save
    testbatch_ID = 0

    # Testing loop
    model.eval()
    test_rcrd_filename = test_rcrd_filename.replace("<epochIDX>", f"{epochIDX:04d}")

    with open(test_rcrd_filename, "a") as test_rcrd_file:
        with torch.no_grad():
            for testbatch_ID, testdata in enumerate(testing_data):
                # Stop when number of testing batches is reached
                if testbatch_ID >= num_test_batches:
                    break

                # Perform a single test step
                noise_gt, noise_pred, test_losses = eval_diffusion_loderunner_datastep(
                    testdata,
                    model,
                    loss_fn,
                    device,
                    in_vars,
                    out_vars,
                )

                # Save testing record
                batch_records = np.column_stack(
                    [
                        np.full(len(test_losses), epochIDX),
                        np.full(len(test_losses), testbatch_ID),
                        test_losses.cpu().detach().numpy().flatten(),
                    ]
                )
                np.savetxt(test_rcrd_file, batch_records, fmt="%d, %d, %.8f")


################################################
# Testing Epoch Functions with Real Data
################################################
if __name__ == "__main__":
    """Test the diffusion epoch functions with real dataset."""
    import argparse
    import sys
    from pathlib import Path
    from torch.utils.data import DataLoader, Subset

    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

    from yoke.models.vit.swin.diffusion_bomberman import DiffusionLodeRunner
    from yoke.datasets.diffusion_dataset import DiffusionLSC_temporal_DataSet

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Test DiffusionLodeRunner epoch functions with real data"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing NPZ files",
    )
    parser.add_argument(
        "--file_prefix_list",
        type=str,
        required=True,
        help="Text file with list of file prefixes",
    )
    parser.add_argument(
        "--max_timeIDX_offset",
        type=int,
        default=10,
        help="Maximum time index offset (default: 10)",
    )
    parser.add_argument(
        "--max_file_checks",
        type=int,
        default=100,
        help="Maximum file check attempts (default: 100)",
    )
    parser.add_argument(
        "--half_image",
        action="store_true",
        help="Use half images (no reflection)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training/validation (default: 2)",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=10,
        help="Number of batches to test per epoch (default: 10)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=2,
        help="Number of epochs to test (default: 2)",
    )
    parser.add_argument(
        "--train_per_val",
        type=int,
        default=1,
        help="Validate every N epochs (default: 1)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_output",
        help="Directory to save training records (default: ./test_output)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Testing DiffusionLodeRunner Epoch Functions")
    print("=" * 60)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Define variables for LSC dataset
    in_vars_names = np.array(
        [
            "density_case",
            "density_cushion",
            "density_maincharge",
            "density_outside_air",
            "density_striker",
            "density_throw",
            "Uvelocity",
            "Wvelocity",
        ]
    )
    out_vars_names = in_vars_names  # Same variables for input and output

    # Create datasets
    print("\nCreating training dataset...")
    train_dataset = DiffusionLSC_temporal_DataSet(
        LSC_NPZ_DIR=args.data_dir,
        file_prefix_list=args.file_prefix_list,
        max_timeIDX_offset=args.max_timeIDX_offset,
        max_file_checks=args.max_file_checks,
        half_image=args.half_image,
        in_vars=in_vars_names,
        out_vars=out_vars_names,
    )
    print(f"Training dataset created with {train_dataset.Nsamples} file prefixes")

    print("\nCreating validation dataset...")
    val_dataset = DiffusionLSC_temporal_DataSet(
        LSC_NPZ_DIR=args.data_dir,
        file_prefix_list=args.file_prefix_list,
        max_timeIDX_offset=args.max_timeIDX_offset,
        max_file_checks=args.max_file_checks,
        half_image=args.half_image,
        in_vars=in_vars_names,
        out_vars=out_vars_names,
    )
    print(f"Validation dataset created with {val_dataset.Nsamples} file prefixes")

    # Printing number of batches in training and validation datasets
    print(f"\nNumber of batches per epoch: {args.num_batches}")
    train_subset = Subset(train_dataset, list(range(args.num_batches * args.batch_size)))
    val_subset = Subset(val_dataset, list(range(args.num_batches * args.batch_size)))

    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Get a sample to determine image dimensions
    print("\nLoading sample to determine dimensions...")
    sample_x, sample_y_tau, sample_noise, sample_lead_time, sample_tau = train_dataset[0]
    height, width = sample_x.shape[1], sample_x.shape[2]
    in_channels = sample_x.shape[0]
    out_channels = sample_y_tau.shape[0]
    print(f"Image dimensions: {height}x{width}")
    print(f"Input channels: {in_channels}")
    print(f"Output channels: {out_channels}")

    # Variable indices (all 8 variables: 0-7)
    in_vars = torch.tensor(list(range(in_channels)))
    out_vars = torch.tensor(list(range(out_channels)))

    # Create model
    print("\nCreating DiffusionLodeRunner model...")
    model = DiffusionLodeRunner(
        default_vars=list(in_vars_names),
        image_size=(height, width),
        patch_size=(10, 10),
        embed_dim=96,
        emb_factor=2,
        num_heads=8,
        block_structure=(1, 1, 3, 1),
        window_sizes=[(8, 8), (8, 8), (4, 4), (2, 2)],
        patch_merge_scales=[(2, 2), (2, 2), (2, 2)],
        verbose=False,
    ).to(device)
    print("Model created successfully")

    # Create optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss(reduction="none")

    # Test simple epoch training
    print("\n" + "=" * 60)
    print("Testing train_simple_diffusion_loderunner_epoch")
    print("=" * 60)

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")

        train_rcrd_filename = str(output_dir / "train_simple_epoch<epochIDX>.csv")
        val_rcrd_filename = str(output_dir / "val_simple_epoch<epochIDX>.csv")

        train_simple_diffusion_loderunner_epoch(
            training_data=train_loader,
            validation_data=val_loader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochIDX=epoch,
            train_per_val=args.train_per_val,
            train_rcrd_filename=train_rcrd_filename,
            val_rcrd_filename=val_rcrd_filename,
            device=device,
            in_vars=in_vars,
            out_vars=out_vars,
            verbose=args.verbose,
        )

        # Check if files were created
        train_file = Path(train_rcrd_filename.replace("<epochIDX>", f"{epoch:04d}"))
        if train_file.exists():
            print(f"  Training record saved to: {train_file}")
            # Read and display summary
            train_data = np.loadtxt(train_file, delimiter=",")
            if len(train_data.shape) == 1:
                train_data = train_data.reshape(1, -1)
            print(f"  Training batches: {len(train_data)}")
            print(f"  Mean training loss: {train_data[:, 2].mean():.6f}")

        if epoch % args.train_per_val == 0:
            val_file = Path(val_rcrd_filename.replace("<epochIDX>", f"{epoch:04d}"))
            if val_file.exists():
                print(f"  Validation record saved to: {val_file}")
                # Read and display summary
                val_data = np.loadtxt(val_file, delimiter=",")
                if len(val_data.shape) == 1:
                    val_data = val_data.reshape(1, -1)
                print(f"  Validation batches: {len(val_data)}")
                print(f"  Mean validation loss: {val_data[:, 2].mean():.6f}")

    # Test evaluation epoch
    print("\n" + "=" * 60)
    print("Testing eval_diffusion_loderunner_epoch")
    print("=" * 60)

    test_rcrd_filename = str(output_dir / "test_epoch<epochIDX>.csv")
    test_epoch_idx = 0

    eval_diffusion_loderunner_epoch(
        testing_data=val_loader,
        num_test_batches=5,  # Limit to 5 batches for testing
        model=model,
        loss_fn=loss_fn,
        epochIDX=test_epoch_idx,
        test_rcrd_filename=test_rcrd_filename,
        device=device,
        in_vars=in_vars,
        out_vars=out_vars,
    )

    test_file = Path(test_rcrd_filename.replace("<epochIDX>", f"{test_epoch_idx:04d}"))
    if test_file.exists():
        print(f"\nTest record saved to: {test_file}")
        # Read and display summary
        test_data = np.loadtxt(test_file, delimiter=",")
        if len(test_data.shape) == 1:
            test_data = test_data.reshape(1, -1)
        print(f"Test batches: {len(test_data)}")
        print(f"Mean test loss: {test_data[:, 2].mean():.6f}")

    print("\n" + "=" * 60)
    print("All epoch tests completed successfully!")
    print("=" * 60)
    print(f"\nOutput files saved to: {output_dir}")
    print("\nFiles created:")
    for csv_file in sorted(output_dir.glob("*.csv")):
        print(f"  - {csv_file.name}")
