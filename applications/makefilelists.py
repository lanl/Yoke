"""Defines functions to make file list.

If run, will make file lists to that specification

Created for use with the Nested Cylinder dataset.

"""

####################################
# Packages
####################################
import os
import glob
import random
import typing
import argparse
import numpy as np

NoneStr = typing.Union[None, str]


####################################
# Make File List Functions
####################################
def findcorruptedfiles(input_dir: str, samplelist: list[str]):
    """Function to identify which npz files in a list have been corrupted

    Args:
        input_dir (str): The directory path where all of the .npz files are
                         stored
        samplelist (list[str]): list of npz files to test

    Returns:
        corrupted (list[str]): list of npz files that were found to be
                               corrupted

    """
    corrupted = []
    for sample in samplelist:
        filepath = os.path.join(input_dir, sample)
        try:
            npz = np.load(filepath)
        except:
            corrupted.append(filepath)

    print(
        len(corrupted),
        "corrupted samples found out of",
        len(samplelist),
        "files tested.",
    )

    return corrupted


def maketvtlists(
    search_dir: str,
    sample_split: tuple[float, float, float],
    save_path: NoneStr = None,
    save: bool = True,
):
    """Function to make training, validation, and testing samples lists and save
    them to files

    Args:
        search_dir (str): file path to directory to search for samples in;
                          include any restrictions for file name
        sample_split (tuple[float, float, float]): training, validation,
                                                   and testing split percentages;
                                                   must sum to 1.0; to create one
                                                   list containing all samples,
                                                   use (1, 0, 0)
        save_path (None or str): path to save .txt file contianing list of samples
        save (bool): boolean for if the sample list is saved to a .txt file

    Returns:
        train_samples (list[str]): list of samples corresponding to sample_split[0]
                                   fraction of total samples; if save=True, will be
                                   saved to .txt file
        val_samples (list[str]): list of samples corresponding to sample_split[1]
                                 fraction of total samples; if save=True, will be
                                 saved to .txt file
        test_samples (list[str]): list of samples corresponding to sample_split[2]
                                  fraction of total samples; if save=True, will be
                                  saved to .txt file

    """
    # Test Sample Split
    assert_str = (
        "Sum of training, validation, and testing split must be less "
        "than or equal to 1.0"
    )
    assert sum(sample_split) <= 1, assert_str

    # Gather Samples
    sample_list = glob.glob(search_dir)
    sample_list = np.unique(sample_list).tolist()
    corrupted = findcorruptedfiles(
        input_dir=os.path.dirname(search_dir), samplelist=sample_list
    )
    for corrfile in corrupted:
        try:
            sample_list.remove(corrfile)
        except:
            pass
    random.shuffle(sample_list)
    total_samples = len(sample_list)

    # Find Split Points
    train, val, test = sample_split
    trainIDX = int(np.floor(train * total_samples))
    valIDX = int(trainIDX + np.floor(val * total_samples))
    testIDX = int(valIDX + np.floor(test * total_samples))

    # Split Sample List
    train_samples = sample_list[:trainIDX]
    val_samples = sample_list[trainIDX:valIDX]
    test_samples = sample_list[valIDX:testIDX]

    # Save to File
    if save:
        if save_path == None:
            raise ValueError(
                "None is not a valid save path for makefilelist. "
                "Either provide a valid save path or use save=False."
            )
        else:
            if train > 0:
                sample_file = open(save_path + "_train_samples.txt", "w")
                np.savetxt(sample_file, train_samples, fmt="%s")
                sample_file.close()

            if val > 0:
                sample_file = open(save_path + "_val_samples.txt", "w")
                np.savetxt(sample_file, val_samples, fmt="%s")
                sample_file.close()

            if test > 0:
                sample_file = open(save_path + "_test_samples.txt", "w")
                np.savetxt(sample_file, test_samples, fmt="%s")
                sample_file.close()

    return train_samples, val_samples, test_samples


########################################################################
########################################################################
descr_str = "Makes filelists for training, validation, and testing"
parser = argparse.ArgumentParser(
    prog="Make file lists", description=descr_str, fromfile_prefix_chars="@"
)
########################
# File Paths
########################
parser.add_argument(
    "--save_dir",
    action="store",
    type=str,
    default="./",
    help="What directory to save model files in",
)

parser.add_argument(
    "--input_dir",
    action="store",
    type=str,
    default="./",
    help="Where the data files are located",
)

parser.add_argument(
    "--data_tag",
    action="store",
    type=str,
    default="nc*pvi*.npz",
    help="Naming convention for the data files",
)
########################
# Data Split
########################
parser.add_argument(
    "--train_split",
    action="store",
    type=float,
    default=0.4,
    help="Percentage of total avialable data to use for training",
)

parser.add_argument(
    "--validation_split",
    action="store",
    type=float,
    default=0.4,
    help="Percentage of total avialable data to use for validation",
)

parser.add_argument(
    "--test_split",
    action="store",
    type=float,
    default=0.2,
    help="Percentage of total avialable data to use for testing",
)

####################################
####################################
if __name__ == "__main__":
    ########################
    # Process Inputs
    ########################
    args = parser.parse_args()

    # File Paths
    save_dir = args.save_dir
    input_dir = args.input_dir
    data_tag = args.data_tag

    # Data Split
    train = args.train_split
    val = args.validation_split
    test = args.test_split

    ########################
    # Make File Lists
    ########################
    TVTsplit = maketvtlists(
        search_dir=os.path.join(input_dir, data_tag),
        sample_split=(train, val, test),
        save_path=os.path.join(save_dir, "datalist"),
        save=True,
    )
    train_samples, val_samples, test_samples = TVTsplit

    ########################
    # Print Information
    ########################
    train_filelist = os.path.join(save_dir, "datalist") + "_train_samples.txt"
    val_filelist = os.path.join(save_dir, "datalist") + "_val_samples.txt"
    test_filelist = os.path.join(save_dir, "datalist") + "_test_samples.txt"

    print(
        "Training filelist created at "
        + train_filelist
        + "\n\tcontaining "
        + str(len(train_samples))
        + " samples,\n\t"
        + str(train * 100)
        + "% of total samples."
    )
    print(
        "Validation filelist created at "
        + val_filelist
        + "\n\tcontaining "
        + str(len(val_samples))
        + " samples,\n\t"
        + str(val * 100)
        + "% of total samples."
    )
    print(
        "Testing filelist created at "
        + test_filelist
        + "\n\tcontaining "
        + str(len(test_samples))
        + " samples,\n\t"
        + str(test * 100)
        + "% of total samples."
    )
