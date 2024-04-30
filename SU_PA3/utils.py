#required imports
import os
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
import torch.nn.functional as F
from torch import Tensor
import librosa
import numpy as np

#padding function
def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

#custom dataset class
class AudioDataset(Dataset):
    def __init__(self, root_dir):
        """
        Initialize the dataset.

        Args:
            root_dir (str): Path to the root directory of the dataset.
        """
        self.root_dir = root_dir
        self.cropLength = 16000
        self.cut=64600
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        """
        Load audio samples and their labels from the dataset directory.
        """
        # Define class directories
        classes = ['Real', 'Fake']

        # Iterate through classes
        for idx, class_name in enumerate(classes):
            class_dir = os.path.join(self.root_dir, class_name)
            for filename in os.listdir(class_dir):
                # Construct the full file path
                file_path = os.path.join(class_dir, filename)
                # Add the sample and label to the list
                if file_path[-3:] == "wav":
                    self.samples.append((file_path, idx))
                elif file_path[:-3] == "mp3":
                    self.samples.append((file_path, idx))
                else:
                    pass

    def __len__(self):
        """
        Return the total number of samples.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a sample and its label at the specified index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing the audio sample and its label.
        """
        file_path, label = self.samples[idx]


        X, fs = torchaudio.load(file_path)
        X = np.array(X[0])
        X_pad = pad(X,self.cut)
        audio = Tensor(X_pad)

        return audio, label

#FOR dataset class
class FORDataset(Dataset):
    def __init__(self, root_dir, mode):
        """
        Initialize the dataset.

        Args:
            root_dir (str): Path to the root directory of the dataset.
            mode (str): Mode of the dataset (e.g., 'training', 'testing', 'validation').
            transform (callable, optional): Optional transform to apply to each audio sample.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.samples = []

        self.cut=64600

        # Define class directories within the specified mode
        self.real_dir = os.path.join(root_dir, mode, 'real')
        self.fake_dir = os.path.join(root_dir, mode, 'fake')

        # Load samples from 'real' and 'fake' directories
        self._load_samples()

    def _load_samples(self):
        """
        Load audio samples and their labels from the dataset directories.
        """
        # Load real samples with label 0
        for filename in os.listdir(self.real_dir):
            file_path = os.path.join(self.real_dir, filename)
            self.samples.append((file_path, 0))

        # Load fake samples with label 1
        for filename in os.listdir(self.fake_dir):
            file_path = os.path.join(self.fake_dir, filename)
            self.samples.append((file_path, 1))

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a sample and its label at the specified index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing the audio sample and its label.
        """
        file_path, label = self.samples[idx]
        
        # Load the audio sample
        X, fs = torchaudio.load(file_path)
        X = np.array(X[0])
        X_pad = pad(X,self.cut)
        audio = Tensor(X_pad)

        return audio, label

# # Example usage
# # Define the root directory where the dataset is stored
# root_dir = "/DATA/arora8/SU/PA3/data/Dataset_Speech_Assignment"

# # Create an instance of the AudioDataset
# audio_dataset = AudioDataset(root_dir)

# # Create a DataLoader for the dataset
# data_loader = DataLoader(audio_dataset, batch_size=32, shuffle=True)

# # Iterate through the data loader
# for batch_idx, (audio_batch, label_batch) in enumerate(data_loader):
#     # Do something with the audio_batch and label_batch
#     print(f"Batch {batch_idx}:")
#     print(f"Audio batch shape: {audio_batch.shape}")
#     print(f"Label batch shape: {label_batch.shape}")

