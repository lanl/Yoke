"""Training for TCNN mapping LSC geometry to density image.

Actual training workhorse for Transpose CNN network mapping layered shaped
charge geometry parameters to density image.

In this version we pass in the directory where the LSC data is stored, so
different drives can be used for different training jobs, and we use a
learning-rate scheduler.

"""

#############################################
# Packages
#############################################
import os
import time
import argparse
import torch
import torch.nn as nn

from yoke.models.surrogateCNNmodules import tCNNsurrogate
# from yoke.datasets.lsc_dataset import LSC_cntr2rho_DataSet
from yoke.datasets.lsc_dataset import LSC_cntr2hfield_DataSet  # SH
import yoke.torch_training_utils as tr
from yoke.helpers import cli
# from yoke.lr_schedulers import CosineWithWarmupScheduler


#  Smyther
def mirror_transform(data):
    geom_params, hfield = data
    # Double the last dimension by concatenating its mirrored version
    hfield = torch.cat((hfield, torch.flip(hfield, dims=[-1])), dim=-1)
    return geom_params, hfield

from torch.utils.data import Dataset

class ButterfliedDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        data = self.base_dataset[index]
        return self.transform(data)

    def __len__(self):
        return len(self.base_dataset)
# -------


#############################################
# Inputs
#############################################
descr_str = (
    "Trains Transpose-CNN to reconstruct density field of LSC simulation "
    "from contours and simulation time. This training uses network with "
    "no interpolation."
)
parser = argparse.ArgumentParser(
    prog="LSC Surrogate Training", description=descr_str, fromfile_prefix_chars="@"
)
parser = cli.add_default_args(parser=parser)
parser = cli.add_filepath_args(parser=parser)
parser = cli.add_computing_args(parser=parser)
parser = cli.add_model_args(parser=parser)
parser = cli.add_training_args(parser=parser)
parser = cli.add_step_lr_scheduler_args(parser=parser)  # SH - kept this for init_learnrate
# parser = cli.add_cosine_lr_scheduler_args(parser=parser)

# Change some default filepaths.
parser.set_defaults(design_file="design_lsc240420_MASTER.csv")


#############################################
#############################################
if __name__ == "__main__":
    #############################################
    # Process Inputs
    #############################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Study ID
    studyIDX = args.studyIDX

    # Data Paths
    design_file = os.path.abspath(args.LSC_DESIGN_DIR + args.design_file)
    train_filelist = args.FILELIST_DIR + args.train_filelist
    validation_filelist = args.FILELIST_DIR + args.validation_filelist
    test_filelist = args.FILELIST_DIR + args.test_filelist

    # Model Parameters
    featureList = args.featureList
    linearFeatures = args.linearFeatures

    # Training Parameters
    initial_learningrate = args.init_learnrate
    batch_size = args.batch_size


    # Leave one CPU out of the worker queue. Not sure if this is necessary.
    num_workers = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])  # - 1
    train_per_val = args.TRAIN_PER_VAL

    # Epoch Parameters
    total_epochs = args.total_epochs
    cycle_epochs = args.cycle_epochs
    train_batches = args.train_batches
    val_batches = args.val_batches
    trn_rcrd_filename = args.trn_rcrd_filename
    val_rcrd_filename = args.val_rcrd_filename
    CONTINUATION = args.continuation
    START = not CONTINUATION
    checkpoint = args.checkpoint

    #############################################
    # Check Devices
    #############################################
    print("\n")
    print("Slurm & Device Information")
    print("=========================================")
    print("Slurm Job ID:", os.environ["SLURM_JOB_ID"])
    print("Pytorch Cuda Available:", torch.cuda.is_available())
    print("GPU ID:", os.environ["SLURM_JOB_GPUS"])
    print("Number of System CPUs:", os.cpu_count())
    print("Number of CPUs per GPU:", os.environ["SLURM_JOB_CPUS_PER_NODE"])

    print("\n")
    print("Model Training Information")
    print("=========================================")

    #############################################
    # Initialize Model
    #############################################

    model = tCNNsurrogate(
        input_size=28,  # SH
        linear_features=(7, 5, linearFeatures),
        initial_tconv_kernel=(5, 5),
        initial_tconv_stride=(5, 5),
        initial_tconv_padding=(0, 0),
        initial_tconv_outpadding=(0, 0),
        initial_tconv_dilation=(1, 1),
        kernel=(3, 3),
        nfeature_list=featureList,
        output_image_size=(1120, 800),
        act_layer=nn.GELU,
    )

    # Wait to move model to GPU until after the checkpoint load. Then
    # explicitly move model and optimizer state to GPU.

    #############################################
    # Initialize Optimizer
    #############################################
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=initial_learningrate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01,
    )

    #############################################
    # Initialize Loss
    #############################################
    # Use `reduction='none'` so loss on each sample in batch can be recorded.
    loss_fn = nn.MSELoss(reduction="none")

    print("Model initialized.")

    #############################################
    # Load Model for Continuation
    #############################################
    if CONTINUATION:
        starting_epoch = tr.load_model_and_optimizer_hdf5(model, optimizer, checkpoint)
        print("Model state loaded for continuation.")
    else:
        starting_epoch = 0

    #############################################
    # Move model and optimizer state to GPU
    #############################################
    if args.multigpu:
        model = nn.DataParallel(model)

    model.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


    #############################################
    # Script and compile model on device
    #############################################
    if args.multigpu:
        compiled_model = model  # jit compilation disabled in multi-gpu scenarios.
    else:
        scripted_model = torch.jit.script(model)

        # Model compilation has some interesting parameters to play with.
        #
        # NOTE: Compiled model is not able to be loaded from checkpoint for some
        # reason.
        compiled_model = torch.compile(
            scripted_model,
            fullgraph=True,  # If TRUE, throw error if
            # whole graph is not
            # compileable.
            mode="reduce-overhead",
        )  # Other compile
        # modes that may
        # provide better
        # performance

    #############################################
    # Initialize Data
    #############################################
    train_dataset = LSC_cntr2hfield_DataSet(args.LSC_NPZ_DIR, train_filelist, design_file)
    val_dataset = LSC_cntr2hfield_DataSet(
        args.LSC_NPZ_DIR, validation_filelist, design_file
    )
    test_dataset = LSC_cntr2hfield_DataSet(args.LSC_NPZ_DIR, test_filelist, design_file)


    train_dataset = ButterfliedDataset(train_dataset, mirror_transform)
    val_dataset = ButterfliedDataset(val_dataset, mirror_transform)
    test_dataset = ButterfliedDataset(test_dataset, mirror_transform)
    # SH - changed dataset

    print("Datasets initialized.")

    #############################################
    # Training Loop
    #############################################
    # Train Model
    print("Training Model . . .")
    starting_epoch += 1
    ending_epoch = min(starting_epoch + cycle_epochs, total_epochs + 1)

    # Setup Dataloaders
    train_dataloader = tr.make_dataloader(
        train_dataset, batch_size, train_batches, num_workers=num_workers
    )
    val_dataloader = tr.make_dataloader(
        val_dataset, batch_size, val_batches, num_workers=num_workers
    )

    for epochIDX in range(starting_epoch, ending_epoch):
        # Time each epoch and print to stdout
        startTime = time.time()

        # Train an Epoch
        tr.train_array_csv_epoch(
            training_data=train_dataloader,
            validation_data=val_dataloader,
            model=compiled_model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochIDX=epochIDX,
            train_per_val=train_per_val,
            train_rcrd_filename=trn_rcrd_filename,
            val_rcrd_filename=val_rcrd_filename,
            device=device,
        )


        endTime = time.time()
        epoch_time = (endTime - startTime) / 60

        # Print Summary Results
        print("Completed epoch " + str(epochIDX) + "...")
        print("Epoch time:", epoch_time)

    # Save Model Checkpoint
    print("Saving model checkpoint at end of epoch " + str(epochIDX) + ". . .")

    # Move the model back to CPU prior to saving to increase portability
    compiled_model.to("cpu")
    # Move optimizer state back to CPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to("cpu")

    # Save model and optimizer state in hdf5
    h5_name_str = "study{0:03d}_modelState_epoch{1:04d}.hdf5"
    new_h5_path = os.path.join("./", h5_name_str.format(studyIDX, epochIDX))
    tr.save_model_and_optimizer_hdf5(
        compiled_model, optimizer, epochIDX, new_h5_path, compiled=True
    )

    #############################################
    # Continue if Necessary
    #############################################
    FINISHED_TRAINING = epochIDX + 1 > total_epochs
    if not FINISHED_TRAINING:
        new_slurm_file = tr.continuation_setup(
            new_h5_path, studyIDX, last_epoch=epochIDX
        )
        os.system(f"sbatch {new_slurm_file}")

    ###########################################################################
    # For array prediction, especially large array prediction, the network is
    # not evaluated on the test set after training. This is performed using
    # the *evaluation* module as a separate post-analysis step.
    ###########################################################################
