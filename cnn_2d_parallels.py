import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

import pandas as pd
import numpy as np

from tqdm import tqdm
import logging
import argparse
import pickle
import os
import sys
from datetime import datetime
import time

import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import classification_report

import yaml

START_DATETIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


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


def multi_acc(y_pred, y_test):
    y_pred_softmax = F.softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    _, y_test_tags = torch.max(y_test, dim=1)

    correct_pred = (y_pred_tags == y_test_tags).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return y_pred_softmax, acc

def train(import_dict, n_epochs, batch_size, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    model.to(device)

    # Load train and test from import_list

    X_train = import_dict["X_train"]
    y_train = import_dict["y_train_labelled"]

    # Load train and test in CUDA Memory
    x_train = torch.tensor(X_train.values.tolist(), dtype=torch.float).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float).to(device)
    
    # Create Torch datasets
    train = torch.utils.data.TensorDataset(x_train, y_train_tensor)

    # Create Data Loaders
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )

    train_loss = []

    for epoch in range(n_epochs):
        start_time = time.time()
        # Set model to train configuration
        model.train()
        avg_loss = 0.0
        accuracy = []
        for i, (x_batch, y_batch) in enumerate(train_loader):
            # Predict/Forward Pass
            y_pred = model(x_batch)
            # # Casting
            # x_batch = x_batch.to(device)
            # y_batch = y_batch.type(torch.LongTensor)
            # y_batch = y_batch.to(device)
            # Compute loss
            loss = loss_fn(y_pred, y_batch)
            _, acc = multi_acc(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
            accuracy.append(acc.item())


        # Check Accuracy
        # val_accuracy = sum(val_preds.argmax(axis=1) == y_test) / len(y_test)
        train_loss.append(avg_loss)
        elapsed_time = time.time() - start_time
        logging.info(
            "Epoch {}/{} \t loss={:.4f} \t acc={:.2f}% \t time={:.2f}s".format(
                epoch + 1,
                n_epochs,
                avg_loss,
                np.mean(accuracy),
                elapsed_time,
            )
        )

    return model


def train_with_val(import_dict, n_epochs, batch_size, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    model.to(device)

    # Load train and test from import_list

    X_train = import_dict["X_train"]
    X_test = import_dict["X_test"]
    y_train = import_dict["y_train_labelled"]
    y_test = import_dict["y_test_labelled"]

    # Load train and test in CUDA Memory
    x_train = torch.tensor(X_train.values.tolist(), dtype=torch.float).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float).to(device)
    x_cv = torch.tensor(X_test.values.tolist(), dtype=torch.float).to(device)
    y_cv = torch.tensor(y_test, dtype=torch.float).to(device)

    # Create Torch datasets
    train = torch.utils.data.TensorDataset(x_train, y_train_tensor)
    valid = torch.utils.data.TensorDataset(x_cv, y_cv)

    # Create Data Loaders
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid, batch_size=batch_size, shuffle=False
    )

    train_loss = []
    valid_loss = []

    for epoch in range(n_epochs):
        start_time = time.time()
        # Set model to train configuration
        model.train()
        avg_loss = 0.0
        accuracy = []
        for i, (x_batch, y_batch) in enumerate(train_loader):
            # Predict/Forward Pass
            y_pred = model(x_batch)
            # # Casting
            # x_batch = x_batch.to(device)
            # y_batch = y_batch.type(torch.LongTensor)
            # y_batch = y_batch.to(device)
            # Compute loss
            loss = loss_fn(y_pred, y_batch)
            _, acc = multi_acc(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
            accuracy.append(acc.item())

        # Set model to validation configuration -Doesn't get trained here
        model.eval()
        avg_val_loss = 0.0
        val_accuracy = []
        val_preds = np.zeros((len(x_cv), 10))

        for i, (x_batch, y_batch) in enumerate(valid_loader):
            # Casting
            # x_batch = x_batch.to(device)
            # y_batch = y_batch.type(torch.LongTensor)
            # y_batch = y_batch.to(device)
            # Detach
            y_pred = model(x_batch).detach()
            val_pred, val_acc = multi_acc(y_pred, y_batch)
            val_preds[i * batch_size : (i + 1) * batch_size] = val_pred.cpu()
            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
            val_accuracy.append(val_acc.item())

        # Check Accuracy
        # val_accuracy = sum(val_preds.argmax(axis=1) == y_test) / len(y_test)
        train_loss.append(avg_loss)
        valid_loss.append(avg_val_loss)
        elapsed_time = time.time() - start_time
        logging.info(
            "Epoch {}/{} \t loss={:.4f} \t acc={:.2f}% \t val_loss={:.4f} \t val_acc={:.2f}% \t time={:.2f}s".format(
                epoch + 1,
                n_epochs,
                avg_loss,
                np.mean(accuracy),
                avg_val_loss,
                np.mean(val_accuracy),
                elapsed_time,
            )
        )

    if not os.path.isdir("Figures"):
        os.mkdir("Figures")

    plot_graph(n_epochs, train_loss, valid_loss)
    plot_confusion(import_dict["y_test"], val_preds)

    return model


def open_pickle(file_path):
    """Unpickles processed data.

    Args:
        file_path (str): File path with processed data.

    Returns:
        dict : Processed data list.
    """
    infile = open(file_path, "rb")
    import_dict = pickle.load(infile)
    infile.close()

    return import_dict


def plot_graph(epochs, train_loss, valid_loss):
    plt.figure(figsize=(12, 12))
    sns.set_style("ticks")
    plt.style.library["seaborn-colorblind"]  # Care for the color-blind :")
    plt.title("Train/Validation Loss")
    plt.plot(list(np.arange(epochs) + 1), train_loss, label="train")
    plt.plot(list(np.arange(epochs) + 1), valid_loss, label="validation")
    plt.xlabel("num_epochs", fontsize=12)
    plt.ylabel("loss", fontsize=12)
    plt.legend(loc="best")
    plt.savefig(fname=f"Figures/{START_DATETIME}_training_with_validation")


def plot_confusion(y_test, val_preds):
    classes = [
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
    y_true = [x for x in y_test.tolist()]
    y_pred = [classes[x] for x in val_preds.argmax(axis=1)]
    skplt.metrics.plot_confusion_matrix(
        y_true, y_pred, figsize=(12, 12), x_tick_rotation=90
    )
    plt.savefig(fname=f"Figures/{START_DATETIME}_confusion_matrix")
    logging.info(f"\n{classification_report(y_true, y_pred)}")

def config_loader(path):
    with open(path, 'r') as stream:
        src_cfgs = yaml.safe_load(stream)
    return src_cfgs

def main():
    if not os.path.isdir("Models"):
        os.mkdir("Models")
        
    if not os.path.isdir("Logs/Models"):
        os.mkdir("Logs/Models")

    # Parser for CLI configs
    parser = argparse.ArgumentParser(description="Training model.")
    parser.add_argument(
        "--processed_path",
        type=str,
        help="Path for the processed data.",
    )
    parser.add_argument(
        "--split",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Whether the data is split into train-test sets.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="Configs/cnn_2d_parallels.yaml",
        help="Configurations for the model.",
    )

    opt = parser.parse_args()
    PATH = opt.processed_path
    SPLIT = opt.split
    CFG = config_loader(opt.config)
    print(CFG)
    n_epochs = CFG["n_epochs"]
    batch_size = CFG["batch_size"]
    lr = CFG["optimiser_cfg"]["lr"]
    
    # Logger configs
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(f"Logs/Models/{START_DATETIME}_cnn_2d_parallels_split_{SPLIT}.log"), logging.StreamHandler(sys.stdout)],
    )

    import_dict = open_pickle(PATH)

    if SPLIT:
        model = train_with_val(import_dict, n_epochs, batch_size, lr)
    else:
        model = train(import_dict, n_epochs, batch_size, lr)
    
    torch.save(model, f"Models/{START_DATETIME}_CNN_2D_Split_{SPLIT}.pt")

if __name__ == '__main__':
    main()
