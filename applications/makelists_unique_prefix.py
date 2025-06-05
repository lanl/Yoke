"""Creates TVT lists of unique *run ID* filename prefixes.

"""

import os
import argparse
import typing
import fnmatch
import random
import numpy as np

NoneStr = typing.Union[None, str]


def find_uniq_prefixes(input_dir: str, prefix_tag: str) -> list:
    """Function to return list of unique prefixes matching a wildcard string.

    Args:
        input_dir (str): Directory to look in for filenames matching prefix wildcard.
        prefix_tag (str): Wildcard representing only the types of filename-prefixes
                          the function should uniquely list.
    
    """

    # Initialize an empty set to store UNIQUE prefixes
    uniq_prefixes = set()
    
    for filename in os.listdir(input_dir):
        # Check if filename matches wildcard prefix
        if fnmatch.fnmatch(filename, prefix_tag+'*'):
            # Extract the prefix
            prefix = filename[:len(prefix_tag)]
            uniq_prefixes.add(prefix)

    # Convert the set to a list
    uniq_prefix_list = list(uniq_prefixes)

    return uniq_prefix_list


def make_prefix_tvtlists(
    prefix_list: list,
    sample_split: tuple[float, float, float],
    save_path: NoneStr = None,
    save: bool = True) -> tuple[list[str], list[str], list[str]]: 
    """Function to make training, validation, and testing sample lists and save
    them to files.

    A main input list of unique filename prefixes is used to divide into TVT lists.

    Args:
        prefix_list (list): List of unique filename prefixes to divide into
                            TVT sub-lists.
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

    random.shuffle(prefix_list)
    total_samples = len(prefix_list)

    # Find Split Points
    train, val, test = sample_split
    trainIDX = int(np.floor(train * total_samples))
    valIDX = int(trainIDX + np.floor(val * total_samples))
    testIDX = int(valIDX + np.floor(test * total_samples))

    # Split Sample List
    train_samples = prefix_list[:trainIDX]
    val_samples = prefix_list[trainIDX:valIDX]
    test_samples = prefix_list[valIDX:testIDX]

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
descr_str = "Makes filelists of unique prefixes for training, validation, and testing"
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
    default="./prefixes",
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
    "--prefix_tag",
    action="store",
    type=str,
    default="lsc240420_id?????",
    help=("Wildcard prefix to use for data files. Must correspond only to the "
          "prefix you want uniquely identified."),
)
########################
# Data Split
########################
parser.add_argument(
    "--train_split",
    action="store",
    type=float,
    default=0.8,
    help="Percentage of total avialable data to use for training",
)

parser.add_argument(
    "--validation_split",
    action="store",
    type=float,
    default=0.1,
    help="Percentage of total avialable data to use for validation",
)

parser.add_argument(
    "--test_split",
    action="store",
    type=float,
    default=0.1,
    help="Percentage of total avialable data to use for testing",
)


if __name__ == "__main__":
    ########################
    # Process Inputs
    ########################
    args = parser.parse_args()

    # File Paths
    save_dir = args.save_dir
    input_dir = args.input_dir
    prefix_tag = args.prefix_tag
    
    # Data Split
    train = args.train_split
    val = args.validation_split
    test = args.test_split

    # Get list of unique prefixes
    uniq_prefix_list = find_uniq_prefixes(input_dir, prefix_tag)
        
    train_list, val_list, test_list = make_prefix_tvtlists(
        uniq_prefix_list,
        sample_split=(train, val, test),
        save_path=save_dir)

    print(f'Train prefixes: {len(train_list)}',
          f'Validation prefixes: {len(val_list)}',
          f'Test prefixes: {len(test_list)}')
