import librosa
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from tqdm import tqdm
import logging
import argparse
import pickle
import os
import sys
from datetime import datetime

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        drp = 0.1
        n_classes = 10

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)

        self.cnn_layer1 = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Dropout(drp),
            # Layer 2
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Dropout(drp),
            # Layer 3
            nn.Conv2d(32, 32, kernel_size=(3, 4), stride=2, padding=0),
            # Out: 62, 29
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 3), stride=2, padding=0),
            # Out: 30, 14
            nn.Dropout(drp),
        )

        self.cnn_layer2 = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=7, stride=1, padding=3),
            nn.Dropout(drp),
            # Layer 2
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
            nn.Dropout(drp),
            # Layer 3
            nn.Conv2d(32, 32, kernel_size=(3, 4), stride=2, padding=0),
            # Out: 62, 29
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 3), stride=2, padding=0),
            # Out: 30, 14
            nn.Dropout(drp),
        )

        self.fc1 = nn.Linear(30 * 14 * 32 * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.out = nn.Linear(64, n_classes)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        out1 = self.cnn_layer1(x)
        out2 = self.cnn_layer2(x)
        full_out = torch.cat([out1, out2], dim=1)
        full_out = full_out.view(full_out.size(0), -1)
        x = torch.flatten(full_out, start_dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.dropout(self.relu(self.fc4(x)))
        x = self.dropout(self.relu(self.fc5(x)))
        out = self.out(x)
        return out


def main():
    # Parser for CLI configs
    parser = argparse.ArgumentParser(description="Cleaning preprocessed data.")
    parser.add_argument(
        "--music_path",
        type=str,
        help="The path to the music file.",
    )
    parser.add_argument(
        "--model_path",
        default="Models/Final_2D_CNN.pt",
        type=str,
        help="The path to the model chosen. Default: `Models/Final_2D_CNN.pt`.",
    )
    
    START_DATETIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if not os.path.isdir("Logs/Predictions"):
        os.mkdir("Logs/Predictions")
    
    # Logger configs
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(f"Logs/Predictions/{START_DATETIME}_predictions.log"), logging.StreamHandler(sys.stdout)],
    )

    opt = parser.parse_args()
    
    sr = 22050
    SAMPLES_PER_SLICE = 63945
    N_MFCC = 60

    logging.info(f"Loading music file from {opt.music_path}...")
    data, sr = librosa.load(opt.music_path)

    mfcc_list = []

    logging.info("Preprocessing...")
    for s in tqdm(range(int(len(data) / SAMPLES_PER_SLICE))):
        start_sample = SAMPLES_PER_SLICE * s
        end_sample = start_sample + SAMPLES_PER_SLICE
        song = data
        mfcc = librosa.feature.mfcc(
            y=song[start_sample:end_sample], sr=sr, n_mfcc=N_MFCC
        )
        mfcc = mfcc.T
        mfcc_list.append(mfcc.tolist())

    logging.info(f"Loading model from {opt.model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(opt.model_path, map_location=device)

    model.eval()

    pred_labels = []

    logging.info("Predicting...")
    for i in tqdm(mfcc_list, total=len(mfcc_list)):
        tensorised = torch.tensor(i)
        output = model(torch.tensor(tensorised.unsqueeze(0)).to(device)).detach()
        pred_label = torch.argmax(F.softmax(output))
        pred_labels.append(pred_label.item())

    genres_list = [
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock",
    ]
    
    pred_cats = []
    
    for i in pred_labels:
        pred_cats.append(genres_list[i])
    
    logging.info(f"\nPredictions for {opt.music_path}:\n{pd.Series(pred_cats).value_counts(normalize=True)}")

if __name__ == '__main__':
    main()