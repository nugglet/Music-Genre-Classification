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

from re import S
import pandas as pd
import librosa
from tqdm import tqdm
import logging
import argparse
import pickle
import os
import sys
from datetime import datetime


def time_series(data_all, sr_all, NUM_SLICES, SAMPLES_PER_SLICE, N_MFCC):
    """Default time series MFCC extractor.

    Args:
        data_all (list): Data from loading audio files.
        sr_all (list): Sample rate of all the audio files.

    Returns:
        tuple: List of MFCC data and labels.
    """

    mfcc_list = []
    labels_list = []

    for i in tqdm(range(len(data_all))):
        for s in range(NUM_SLICES):
            start_sample = SAMPLES_PER_SLICE * s
            end_sample = start_sample + SAMPLES_PER_SLICE
            song = data_all[i]
            mfcc = librosa.feature.mfcc(
                y=song[start_sample:end_sample], sr=sr_all[i], n_mfcc=N_MFCC
            )
            mfcc = mfcc.T
            mfcc_list.append(mfcc.tolist())
            labels_list.append(i)
    return mfcc_list, labels_list


def htk(data_all, sr_all, NUM_SLICES, SAMPLES_PER_SLICE, N_MFCC):
    """Default time series MFCC extractor with htk enabled.

    Args:
        data_all (list): Data from loading audio files.
        sr_all (list): Sample rate of all the audio files.

    Returns:
        tuple: List of MFCC data and labels.
    """

    mfcc_list = []
    labels_list = []

    for i in tqdm(range(len(data_all))):
        for s in range(NUM_SLICES):
            start_sample = SAMPLES_PER_SLICE * s
            end_sample = start_sample + SAMPLES_PER_SLICE
            song = data_all[i]
            mfcc = librosa.feature.mfcc(
                y=song[start_sample:end_sample], sr=sr_all[i], n_mfcc=N_MFCC, htk=True
            )
            mfcc = mfcc.T
            mfcc_list.append(mfcc.tolist())
            labels_list.append(i)
    return mfcc_list, labels_list


def main():
    
    if not os.path.isdir("Preprocessed"):
        os.mkdir("Preprocessed")
        
    if not os.path.isdir("Logs/Preprocessed"):
        os.mkdir("Logs/Preprocessed")

    # Parser for CLI configs
    parser = argparse.ArgumentParser(description="MFCC extraction.")
    parser.add_argument(
        "--features_path",
        default="Data/features_30_sec.csv",
        type=str,
        help="Path for the features CSV. Default: 'Data/features_30_sec.csv'",
    )
    parser.add_argument(
        "--folder",
        default="Data/",
        type=str,
        help="Folder containing `genres_original` folder with music inside. Default: 'Data/'",
    )
    parser.add_argument(
        "--mfcc_type",
        default="time-series",
        type=str,
        help="The method used to extract MFCC. Default: time-series. Options: htk",
    )
    parser.add_argument(
        "--total_samples",
        default=29,
        type=int,
        help="Specify total samples. Default: 29.",
    )
    parser.add_argument(
        "--num_slices",
        default=10,
        type=int,
        help="Specify number of slices. Default: 10.",
    )
    parser.add_argument(
        "--n_mfcc",
        default=60,
        type=int,
        help="Specify number of MFCCs to extract. Default: 60.",
    )

    opt = parser.parse_args()
    
    FEATURES_PATH = opt.features_path
    FOLDER = opt.folder

    MFCC_TYPE = opt.mfcc_type

    sr = 22050
    TOTAL_SAMPLES = opt.total_samples * sr
    NUM_SLICES = opt.num_slices
    SAMPLES_PER_SLICE = int(TOTAL_SAMPLES / NUM_SLICES)
    N_MFCC = opt.n_mfcc
    START_DATETIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Logger configs
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(f"Logs/Preprocessed/{START_DATETIME}_mfcc_list_{MFCC_TYPE}_{TOTAL_SAMPLES}_{NUM_SLICES}_{N_MFCC}.log"), logging.StreamHandler(sys.stdout)],
    )

    # Start
    logging.info("Starting MFCC extraction...")
    df = pd.read_csv(FEATURES_PATH)
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
            data, sr = librosa.load(FOLDER + i)
            data_all.append(data)
            sr_all.append(sr)
        except:
            failed_load.append(i)
            logging.warning(f"The file {i} was not loaded.")

    # MFCC extraction
    # ! Are there more options?
    logging.info(f"Converting into MFCC using {MFCC_TYPE}...")

    if MFCC_TYPE == "time-series":
        mfcc_list, labels_list = time_series(
            data_all, sr_all, NUM_SLICES, SAMPLES_PER_SLICE, N_MFCC
        )
    elif MFCC_TYPE == "htk":
        mfcc_list, labels_list = time_series(
            data_all, sr_all, NUM_SLICES, SAMPLES_PER_SLICE, N_MFCC
        )
    else:
        logging.error("MFCC type not specified correctly.")

    new_df = pd.DataFrame({"mfcc": mfcc_list, "labels": labels_list})
    # Pickling
    logging.info(f"Dumping into pickle file...")
    if os.path.isfile(f"Preprocessed/mfcc_list_{MFCC_TYPE}_{TOTAL_SAMPLES}_{NUM_SLICES}_{N_MFCC}"):
        os.remove(f"Preprocessed/mfcc_list_{MFCC_TYPE}_{TOTAL_SAMPLES}_{NUM_SLICES}_{N_MFCC}")

    outfile = open(f"Preprocessed/mfcc_list_{MFCC_TYPE}_{TOTAL_SAMPLES}_{NUM_SLICES}_{N_MFCC}", "wb")
    pickle.dump(new_df, outfile)
    outfile.close()
    logging.info("Done.")


if __name__ == "__main__":
    main()
