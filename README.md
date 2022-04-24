# Music-Genre-Classification
50.039 Theory and Application of Deep Learning: Project

## Usage

Git clone this project. To predict the music genre of your audio file, just use

	$ python predict_audio.py --music_path [music-path] # defaults to use our final model

The outputs will be the classification labels for each predetermined chunk of the music file that is aggregated into its normalised value counts. For example,

```
Predictions for test.mp3:
classical    0.726316
jazz         0.263158
country      0.010526
```

Of course, this means that it is 72.6% percent sure that it is of the classical genre.

## Setup

### Packages required

- Python 3.9 and above
- torch
- sklearn
- pandas
- numpy
- librosa
- tqdm
- seaborn
- scikit-plot
- matplotlib

### Process the data

	$ python [script-name] -h # for help

- [`mfcc_extractor.py`](./mfcc_extractor.py) will extract MFCC features from the dataset.
- [`process.py`](./process.py) will process and split the data into train and test sets if needed.
- [`cnn_2d_parallels.py`](./cnn_2d_parallels.py) will train the model based on the outputs of [`process.py`](./process.py).
- [`predict_audio.py`](./predict_audio.py) will run the model chosen to output a prediction for a music file.
