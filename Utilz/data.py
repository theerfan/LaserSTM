import os

import numpy as np
import torch
from torch.utils import data
import h5py

from typing import Iterable


class CustomSequence(data.Dataset):
    def __init__(
        self,
        data_dir: str,
        file_indexes: Iterable,
        test_mode: bool = False,
        crystal_length: int = 100,
        load_in_gpu: bool = True,
    ):
        self.file_indexes = file_indexes
        self.test_mode = test_mode
        self.data_dir = data_dir
        self.crystal_length = crystal_length
        self.load_in_gpu = load_in_gpu
        self._num_samples_per_file = 10_000

    def load_data_point(self, file_idx, sample_idx):
        with h5py.File(os.path.join(self.data_dir, "X_new_data.h5"), "r") as file:
            dataset = file[f"dataset_{file_idx}"]
            data = dataset[sample_idx]

        with h5py.File(os.path.join(self.data_dir, "y_new_data.h5"), "r") as file:
            dataset = file[f"dataset_{file_idx}"]
            labels = dataset[sample_idx]

        return data, labels

    def __len__(self):
        # Assuming every file has the same number of samples, otherwise you need a more dynamic way
        return len(self.file_indexes) * self._num_samples_per_file

    def __getitem__(self, idx):
        # Compute file index and sample index within that file
        # shift the file index to make sure we're starting from the specified file index
        idx = idx + self._num_samples_per_file * self.file_indexes[0]
        file_idx = idx // self._num_samples_per_file
        sample_idx = idx % self._num_samples_per_file

        data, labels = self.load_data_point(file_idx, sample_idx)
        if self.test_mode:
            data = data[:: self.crystal_length]
            labels = labels[self.crystal_length - 1 : :][:: self.crystal_length]

        data_tensor = torch.tensor(data, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        if self.load_in_gpu:
            # Each file is about 3 GB, just move directly into GPU memory
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            data_tensor = data_tensor.to(device)
            labels_tensor = labels_tensor.to(device)

        return data_tensor, labels_tensor
