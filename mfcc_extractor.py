import pandas as pd
import librosa
from tqdm import tqdm
import logging
import argparse
import pickle
import os
import sys


def time_series(data_all, sr_all):

    mfcc_list = []

    for i in tqdm(range(len(data_all))):
        mfcc = librosa.feature.mfcc(y=data_all[i], sr=sr_all[i])
        mfcc_list.append(mfcc)
    
    return mfcc_list


def htk(data_all, sr_all):
    
    mfcc_list = []

    for i in tqdm(range(len(data_all))):
        mfcc = librosa.feature.mfcc(y=data_all[i], sr=sr_all[i], htk=True)
        mfcc_list.append(mfcc)
    
    return mfcc_list


def log_power(data_all, sr_all):
    
    mfcc_list = []

    for i in tqdm(range(len(data_all))):
        S = librosa.feature.melspectrogram(y=data_all[i], sr=sr_all[i])
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S))
        mfcc_list.append(mfcc)
    
    return mfcc_list


def time_series_hop(data_all, sr_all, hop_length):

    mfcc_list = []

    for i in tqdm(range(len(data_all))):
        mfcc = librosa.feature.mfcc(y=data_all[i], sr=sr_all[i], hop_length=hop_length)
        mfcc_list.append(mfcc)
    
    return mfcc_list


def htk_hop(data_all, sr_all, hop_length):
    
    mfcc_list = []

    for i in tqdm(range(len(data_all))):
        mfcc = librosa.feature.mfcc(y=data_all[i], sr=sr_all[i], htk=True, hop_length=hop_length)
        mfcc_list.append(mfcc)
    
    return mfcc_list


def log_power_hop(data_all, sr_all, hop_length):
    
    mfcc_list = []

    for i in tqdm(range(len(data_all))):
        S = librosa.feature.melspectrogram(y=data_all[i], sr=sr_all[i], hop_length=hop_length)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S))
        mfcc_list.append(mfcc)
    
    return mfcc_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MFCC extraction.')
    parser.add_argument('--mfcc_type', default='time-series', type=str, help='The method used to extract MFCC. Default: time-series. Options: htk, log-power')
    parser.add_argument('--hop_length', default=-1, type=int, help='Specify integer hop length. Default: Auto.')

    opt = parser.parse_args()

    mfcc_type = opt.mfcc_type
    hop_length = opt.hop_length

    hop_modified = True

    if hop_length == -1:
        hop_modified = False

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info("Starting mfcc extraction...")
    df = pd.read_csv("Data/features_30_sec.csv")
    logging.info("Processing csv file...")
    df["path"] = "genres_original/" + df["label"] + "/" + df["filename"]

    paths = df["path"].tolist()

    data_all = []
    sr_all = []
    failed_load = []

    logging.info("Loading music data...")
    for i in tqdm(paths):
        try:
            data, sr = librosa.load("Data/" + i)
            data_all.append(data)
            sr_all.append(sr)
        except:
            failed_load.append(i)
            logging.warning(f"The file {i} was not loaded.")


    logging.info(f"Converting into mfcc using {mfcc_type}...")
    if hop_modified:
        if mfcc_type == 'time-series':
            mfcc_list = time_series_hop(data_all, sr_all, hop_length)
        elif mfcc_type == 'htk':
            mfcc_list = htk_hop(data_all, sr_all, hop_length)
        elif mfcc_type == 'log-power':
            mfcc_list = log_power_hop(data_all, sr_all, hop_length)
        else:
            logging.error("MFCC type not specified correctly.")
    else:
        if mfcc_type == 'time-series':
            mfcc_list = time_series(data_all, sr_all)
        elif mfcc_type == 'htk':
            mfcc_list = htk(data_all, sr_all)
        elif mfcc_type == 'log-power':
            mfcc_list = log_power(data_all, sr_all)
        else:
            logging.error("MFCC type not specified correctly.")

    logging.info(f"Dumping into pickle file...")
    if os.path.isfile("mfcc_list"):
        os.remove("mfcc_list")
    
    outfile = open("mfcc_list", 'wb')
    pickle.dump(mfcc_list, outfile)
    outfile.close()
    logging.info("Done.")