import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from tqdm import tqdm
import logging
import argparse
import pickle
import os
import sys
from datetime import datetime


def open_pickle(file_path):
    """Unpickles MFCC data.

    Args:
        file_path (str): File path with MFCC data.

    Returns:
        pandas.DataFrame : DataFrame with MFCC data and labels.
    """
    infile = open(file_path, "rb")
    mfcc_list = pickle.load(infile)
    infile.close()

    return mfcc_list


def get_label_names(mfcc_list, features_data):
    """Get label names for the MFCC data.

    Args:
        mfcc_list (pandas.DataFrame): DataFrame for MFCC.
        features_data (pandas.DataFrame): DataFrame for features.

    Returns:
        Pandas.DataFrame: Combined DataFrame.
    """    
    df = pd.read_csv(features_data)
    df["labels"] = range(0, 1000)
    mfcc_list = mfcc_list.merge(df, on="labels", how="left")

    return mfcc_list


def split_train_test(mfcc_list, test_size=0.2):
    """Train-test split.

    Args:
        mfcc_list (pandas.DataFrame): MFCC DataFrame.
        test_size (float, optional): Test split size. Defaults to 0.2.

    Returns:
        Multiple: Train-test splits.
    """    
    X_train, X_test, y_train, y_test = train_test_split(
        mfcc_list.mfcc, mfcc_list.label, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test


def one_hot_labels(y_train, y_test=None):
    """Using LabelBinarizer to create one-hot labels.

    Args:
        y_train (list): Training labels.
        y_test (list, optional): Testing labels. Defaults to None.

    Returns:
        Multiple: Training and/or testing one-hot labels.
    """    
    le = LabelBinarizer()

    y_train_labelled = le.fit_transform(y_train.values)

    if not isinstance(y_test, type(None)):
        y_test_labelled = le.transform(y_test.values)
        return y_train_labelled, y_test_labelled
    else:
        return y_train_labelled


def main():

    if not os.path.isdir("Processed"):
        os.mkdir("Processed")

    if not os.path.isdir("Logs/Processed"):
        os.mkdir("Logs/Processed")

    # Parser for CLI configs
    parser = argparse.ArgumentParser(description="Cleaning preprocessed data.")
    parser.add_argument(
        "--file_path",
        default="Preprocessed/mfcc_list_time-series_639450_10_60",
        type=str,
        help="The path to the pickled MFCC file. Defaults to 'Preprocessed/mfcc_list_time-series_639450_10_60'",
    )
    parser.add_argument(
        "--features_data",
        default="Data/features_30_sec.csv",
        type=str,
        help="The path to the CSV file containing feature data. Defaults to 'Data/features_30_sec.csv'",
    )
    parser.add_argument(
        "--train_test_split",
        default=0.2,
        type=float,
        help="Specify the train-test split for training. Disables splitting when set as 0.",
    )

    opt = parser.parse_args()

    FILE_PATH = opt.file_path
    FEATURES_DATA = opt.features_data
    SPLIT = opt.train_test_split
    SPLIT_DECISION = "split"

    START_DATETIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if SPLIT == 0:
        SPLIT_DECISION = "no_split"

    # Logger configs
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(
                f"Logs/Processed/{START_DATETIME}_process_{SPLIT_DECISION}_{int(SPLIT * 100)}.log"
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Start
    logging.info(f"Taking MFCC data from {FILE_PATH}...")
    mfcc_list = open_pickle(FILE_PATH)
    logging.info(f"Taking features CSV from {FEATURES_DATA}...")
    mfcc_list = get_label_names(mfcc_list, FEATURES_DATA)
    if SPLIT_DECISION == "split":
        logging.info(f"Splitting into {SPLIT} for test...")
        X_train, X_test, y_train, y_test = split_train_test(mfcc_list, SPLIT)
        y_train_labelled, y_test_labelled = one_hot_labels(y_train, y_test)
        export_list = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_train_labelled": y_train_labelled,
            "y_test_labelled": y_test_labelled,
        }
    else:
        logging.info("Not splitting...")
        X_train = mfcc_list.mfcc
        y_train = mfcc_list.label
        y_train_labelled = one_hot_labels(y_train)
        export_list = {
            "X_train": X_train,
            "y_train": y_train,
            "y_train_labelled": y_train_labelled,
        }

    # Pickling
    logging.info(f"Dumping into pickle file...")
    if os.path.isfile(
        f"Processed/{START_DATETIME}_processed_{SPLIT_DECISION}_{int(SPLIT * 100)}"
    ):
        os.remove(
            f"Processed/{START_DATETIME}_processed_{SPLIT_DECISION}_{int(SPLIT * 100)}"
        )

    outfile = open(
        f"Processed/{START_DATETIME}_processed_{SPLIT_DECISION}_{int(SPLIT * 100)}",
        "wb",
    )
    pickle.dump(export_list, outfile)
    outfile.close()
    logging.info("Done.")

if __name__ == "__main__":
    main()