import os

import numpy as np
import torch
from torch.utils import data
import h5py

from typing import Iterable


class CustomSequence(data.Dataset):
    # Adding prefixes to file names we're trying to iterate
    # because we have multiple versions of the data
    @staticmethod
    def add_prefix(lst: list, prefix="X"):
        return [prefix + "_" + str(i) + ".npy" for i in lst]

    def __init__(
        self,
        data_dir: str,
        file_indexes: Iterable,
        test_mode: bool = False,
        train_prefix: str = "X_new",
        val_prefix: str = "y_new",
        crystal_length: int = 100,
        load_in_gpu: bool = True,
    ):
        self.Xnames = CustomSequence.add_prefix(file_indexes, train_prefix)
        self.ynames = CustomSequence.add_prefix(file_indexes, val_prefix)
        self.file_indexes = file_indexes
        self.test_mode = test_mode
        self.data_dir = data_dir
        self.crystal_length = crystal_length
        self.current_file_idx = None
        self.current_data = None
        self.current_labels = None
        self.load_in_gpu = load_in_gpu

    # NOTE: We always assume that we have 10000 examples per file.
    def get_num_samples_per_file(self):
        return 10_000

    def non_shuffled_load_data_for_file_index(self, file_idx):
        if file_idx != self.current_file_idx:
            # Clear memory of previously loaded data
            self.current_data = None
            self.current_labels = None
            # Load new data
            self.current_data = np.load(
                os.path.join(self.data_dir, self.Xnames[file_idx])
            )
            self.current_labels = np.load(
                os.path.join(self.data_dir, self.ynames[file_idx])
            )
            self.current_file_idx = file_idx
        else:
            pass

    def shuffled_load_data_point(self, file_idx, sample_idx):
        with h5py.File(os.path.join(self.data_dir, "X_new_data.h5"), "r") as file:
            dataset = file[f"dataset_{file_idx}"]
            data = dataset[sample_idx]
        
        with h5py.File(os.path.join(self.data_dir, "y_new_data.h5"), "r") as file:
            dataset = file[f"dataset_{file_idx}"]
            labels = dataset[sample_idx]
        
        return data, labels


    def __len__(self):
        # Assuming every file has the same number of samples, otherwise you need a more dynamic way
        return len(self.Xnames) * self.get_num_samples_per_file()

    # NOTE: We assume that the suffling only happens when we load the data
    # in a non-test mode.
    def __getitem__(self, idx):
        # Compute file index and sample index within that file
        num_samples_per_file = self.get_num_samples_per_file()
        file_idx = idx // num_samples_per_file
        sample_idx = idx % num_samples_per_file

        if self.test_mode:
            # Load data if not already in memory or if file index has changed
            self.non_shuffled_load_data_for_file_index(file_idx)

            data = self.current_data
            labels = self.current_labels

            # In test mode, we only care about the first thing that goes
            # into the crystal and the thing that comes out. (All steps of crystal at once)
            data = data[:: self.crystal_length]
            labels = labels[self.crystal_length - 1 : :][:: self.crystal_length]

            # Get the specific sample from the currently loaded data
            data = self.current_data[sample_idx]
            labels = self.current_labels[sample_idx]
        else:
            data, labels = self.shuffled_load_data_point(file_idx, sample_idx)

        data_tensor = torch.tensor(data, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        if self.load_in_gpu:
            # Each file is about 3 GB, just move directly into GPU memory
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            data_tensor = data_tensor.to(device)
            labels_tensor = labels_tensor.to(device)

        if torch.max(data_tensor) > 10.0:
            print("STOP RIGHT THERE")
            # data_tensor = data_tensor / torch.max(data_tensor)

        return data_tensor, labels_tensor
