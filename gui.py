import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog as fd

import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F

import os

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
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
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


class GUI:
  genres = [
        "Blues",
        "Classical",
        "Country",
        "Disco",
        "Hiphop",
        "Jazz",
        "Metal",
        "Pop",
        "Reggae",
        "Rock",
    ]
  
  def __init__(self, model_path):
    self.model_path = model_path
    self.model = self.load_model()
    self.model.eval()
    
    self.root = tk.Tk()
    self.root.title("Music Genre Classifier")
    self.root.resizable(False, False)
    self.root.geometry('420x210')
    
    self.file_btn = ttk.Button(self.root, text="Choose an Audio File", command=self.predict_file)
    self.header = ttk.Label(self.root, text="Welcome! Please choose an audio file to predict its genre.")
    self.prediction = ttk.Label(self.root, text="No audio sample selected.")
    
    
  def start(self):
    self.header.pack()
    self.file_btn.pack(expand=True)
    self.prediction.pack()
    self.root.mainloop()
  
  def predict_file(self):
    filetypes = (
        ('wav files', '*.wav'),
        ('mp3 files', '*.mp3'),
        ('flac files', '*.flac')
    )

    filename = fd.askopenfilename(
        title='Open a file',
        initialdir='.',
        filetypes=filetypes)
    
    try:
      mfcc_list = self.get_mfcc(filename)
      label = self.predict(mfcc_list)
      print(label)
      self.prediction.config(text=f"{os.path.basename(filename)} is predicted to be in the {label} genre!")
      
    except Exception as e:
      self.prediction.config(text="Invalid file!")
      
  def predict(self, mfcc_list):
    predicted = []
    for i in mfcc_list:
      inp = torch.tensor(i)
      outp = self.model(torch.tensor(inp.unsqueeze(0)).to(self.device)).detach()
      pred_label = torch.argmax(F.softmax(outp))
      predicted.append(pred_label.item())
    
    for i in range(len(predicted)):
      predicted[i] = self.genres[predicted[i]]
    
    return max(predicted, key=predicted.count)
  
  def load_model(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(self.model_path, map_location=self.device)
    return model

  def get_mfcc(self, filename, TOTAL_SAMPLES=29, NUM_SLICES=10, N_MFCC=60, SAMPLE_RATE=22050):
    SAMPLES_PER_SLICE = int(TOTAL_SAMPLES * SAMPLE_RATE / NUM_SLICES)
    song, sample_rate = librosa.load(filename)
    mfcc_list = []

    for s in range(int(len(song)/SAMPLES_PER_SLICE)):
      start_sample = SAMPLES_PER_SLICE * s
      end_sample = start_sample + SAMPLES_PER_SLICE
      mfcc = librosa.feature.mfcc(
          y=song[start_sample:end_sample], sr=sample_rate, n_mfcc=N_MFCC
      )
      mfcc = mfcc.T
      mfcc_list.append(mfcc.tolist())

    return mfcc_list

if __name__ == "__main__":
  gui = GUI("Models/Final_2D_CNN.pt")
  gui.start()