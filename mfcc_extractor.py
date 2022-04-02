#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MFCC extractor.

Extracts MFCC data from the ``GTZAN Dataset - Music Genre Classification``
dataset from Kaggle.

Example:
    To just run the default settings, you can use the CLI.

        $ python mfcc_extractor.py
    
    Alternatively, you can see the different options you can change.
    
        $ python mfcc_extractor.py --help
"""

import pandas as pd
import librosa
from tqdm import tqdm
import logging
import argparse
import pickle
import os
import sys


def time_series(data_all, sr_all):
    """Default time series MFCC extractor.

    Args:
        data_all (list): Data from loading audio files.
        sr_all (list): Sample rate of all the audio files.

    Returns:
        _type_: List of MFCC data.
    """

    mfcc_list = []

    for i in tqdm(range(len(data_all))):
        mfcc = librosa.feature.mfcc(y=data_all[i], sr=sr_all[i])
        mfcc_list.append(mfcc)

    return mfcc_list


def htk(data_all, sr_all):
    """Default time series MFCC extractor with htk enabled.

    Args:
        data_all (list): Data from loading audio files.
        sr_all (list): Sample rate of all the audio files.

    Returns:
        _type_: List of MFCC data.
    """

    mfcc_list = []

    for i in tqdm(range(len(data_all))):
        mfcc = librosa.feature.mfcc(y=data_all[i], sr=sr_all[i], htk=True)
        mfcc_list.append(mfcc)

    return mfcc_list


def log_power(data_all, sr_all):
    """Log-power MFCC extractor.

    Args:
        data_all (list): Data from loading audio files.
        sr_all (list): Sample rate of all the audio files.

    Returns:
        _type_: List of MFCC data.
    """

    mfcc_list = []

    for i in tqdm(range(len(data_all))):
        S = librosa.feature.melspectrogram(y=data_all[i], sr=sr_all[i])
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S))
        mfcc_list.append(mfcc)

    return mfcc_list


def time_series_hop(data_all, sr_all, hop_length):
    """Default time series MFCC extractor with custom hop length.

    Args:
        data_all (list): Data from loading audio files.
        sr_all (list): Sample rate of all the audio files.

    Returns:
        _type_: List of MFCC data.
    """

    mfcc_list = []

    for i in tqdm(range(len(data_all))):
        mfcc = librosa.feature.mfcc(y=data_all[i], sr=sr_all[i], hop_length=hop_length)
        mfcc_list.append(mfcc)

    return mfcc_list


def htk_hop(data_all, sr_all, hop_length):
    """Default time series MFCC extractor with htk enabled and custom hop-length.

    Args:
        data_all (list): Data from loading audio files.
        sr_all (list): Sample rate of all the audio files.

    Returns:
        _type_: List of MFCC data.
    """

    mfcc_list = []

    for i in tqdm(range(len(data_all))):
        mfcc = librosa.feature.mfcc(
            y=data_all[i], sr=sr_all[i], htk=True, hop_length=hop_length
        )
        mfcc_list.append(mfcc)

    return mfcc_list


def log_power_hop(data_all, sr_all, hop_length):
    """Log-power MFCC extractor with custom hop-length.

    Args:
        data_all (list): Data from loading audio files.
        sr_all (list): Sample rate of all the audio files.

    Returns:
        _type_: List of MFCC data.
    """

    mfcc_list = []

    for i in tqdm(range(len(data_all))):
        S = librosa.feature.melspectrogram(
            y=data_all[i], sr=sr_all[i], hop_length=hop_length
        )
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S))
        mfcc_list.append(mfcc)

    return mfcc_list


if __name__ == "__main__":

    # Parser for CLI configs
    parser = argparse.ArgumentParser(description="MFCC extraction.")
    parser.add_argument(
        "--mfcc_type",
        default="time-series",
        type=str,
        help="The method used to extract MFCC. Default: time-series. Options: htk, log-power",
    )
    parser.add_argument(
        "--hop_length",
        default=-1,
        type=int,
        help="Specify integer hop length. Default: Auto.",
    )

    opt = parser.parse_args()

    MFCC_TYPE = opt.mfcc_type
    HOP_LENGTH = opt.hop_length

    hop_modified = True

    if HOP_LENGTH == -1:
        hop_modified = False

    # Logger configs
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("debug.log"), logging.StreamHandler(sys.stdout)],
    )

    # Start
    logging.info("Starting MFCC extraction...")
    df = pd.read_csv("Data/features_30_sec.csv")
    logging.info("Processing csv file...")
    df["path"] = "genres_original/" + df["label"] + "/" + df["filename"]

    paths = df["path"].tolist()

    data_all = []
    sr_all = []
    failed_load = []

    # Loading audio files
    logging.info("Loading music data...")
    for i in tqdm(paths):
        try:
            data, sr = librosa.load("Data/" + i)
            data_all.append(data)
            sr_all.append(sr)
        except:
            failed_load.append(i)
            logging.warning(f"The file {i} was not loaded.")

    # MFCC extraction
    # ! Are there more options?
    logging.info(f"Converting into MFCC using {MFCC_TYPE}...")
    if hop_modified:
        if MFCC_TYPE == "time-series":
            mfcc_list = time_series_hop(data_all, sr_all, HOP_LENGTH)
        elif MFCC_TYPE == "htk":
            mfcc_list = htk_hop(data_all, sr_all, HOP_LENGTH)
        elif MFCC_TYPE == "log-power":
            mfcc_list = log_power_hop(data_all, sr_all, HOP_LENGTH)
        else:
            logging.error("MFCC type not specified correctly.")
    else:
        if MFCC_TYPE == "time-series":
            mfcc_list = time_series(data_all, sr_all)
        elif MFCC_TYPE == "htk":
            mfcc_list = htk(data_all, sr_all)
        elif MFCC_TYPE == "log-power":
            mfcc_list = log_power(data_all, sr_all)
        else:
            logging.error("MFCC type not specified correctly.")

    # Pickling
    logging.info(f"Dumping into pickle file...")
    if os.path.isfile("mfcc_list"):
        os.remove("mfcc_list")

    outfile = open("mfcc_list", "wb")
    pickle.dump(mfcc_list, outfile)
    outfile.close()
    logging.info("Done.")
