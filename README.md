# Music-Genre-Classification
50.039 Theory and Application of Deep Learning: Project

## Setup

### Virtual environments

Due to a conflict over NumPy versions used by `librosa`'s dependency
`Numba` and the latest versions of `PyTorch`, it is recommended that 
the usage of `librosa`-dependent scripts be done on a separate virtual 
environment.

Example (for extracting MFCC from the dataset):

	$ conda create -n dlproject_librosa python=3.9.12
  	$ conda activate dlproject_librosa
	$ conda install pip # in case of a blank environment
	$ pip install jupyter librosa tqdm
	$ python mfcc_extractor.py # --help for help

### Preprocessing the data

- [`mfcc_extractor.py`](./mfcc_extractor.py) will extract MFCC features from the dataset.

### Quirks

- [`jazz.00054.wav`](./Data/genres_original/jazz/jazz.00054.wav) will not be loaded. Possibly is corrupt
