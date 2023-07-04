import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils import data


def add_prefix(lst: list, prefix="X"):
    """
    Add prefix to list of file names
    @param lst: list of file names
    @param prefix: prefix to add
    @return: list of file names with prefix
    """
    return [prefix + "_" + str(i) + ".npy" for i in lst]


class CustomSequence(data.Dataset):
    def __init__(
        self,
        data_dir: str,
        file_idx: list,
        file_batch_size: int,
        model_batch_size: int,
        test_mode: bool = False,
        train_prefix: str = "X_new",
        val_prefix: str = "y_new",
    ):
        """
        Custom PyTorch dataset for loading data
        @param data_dir: directory containing data
        @param file_idx: list of file indices to load
        @param file_batch_size: number of files to load at once
        @param model_batch_size: number of samples to load at once to feed into model
        @param test_mode: whether to load data for testing
        """
        self.Xnames = add_prefix(file_idx, train_prefix)
        self.ynames = add_prefix(file_idx, val_prefix)
        self.file_batch_size = file_batch_size
        self.model_batch_size = model_batch_size
        self.test_mode = test_mode
        self.data_dir = data_dir

    def __len__(self):
        return int(np.ceil(len(self.Xnames) / float(self.file_batch_size)))

    def __getitem__(self, idx):
        batch_x = self.Xnames[
            idx * self.file_batch_size : (idx + 1) * self.file_batch_size
        ]
        batch_y = self.ynames[
            idx * self.file_batch_size : (idx + 1) * self.file_batch_size
        ]

        data = []
        labels = []

        for x, y in zip(batch_x, batch_y):
            if self.test_mode:
                # Every 100th sample
                temp_x = np.load(os.path.join(self.data_dir, x))[::100]
                # Every 100th sample, starting from 100th sample
                temp_y = np.load(os.path.join(self.data_dir, y))[99:][::100]
            else:
                temp_x = np.load(os.path.join(self.data_dir, x))
                temp_y = np.load(os.path.join(self.data_dir, y))

            data.extend(temp_x)
            labels.extend(temp_y)

        for i in range(0, len(data), self.model_batch_size):
            data_batch = data[i : i + self.model_batch_size]
            labels_batch = labels[i : i + self.model_batch_size]

            data_tensor = torch.tensor(data_batch)
            label_tensor = torch.tensor(labels_batch)

            yield data_tensor, label_tensor


def train(
    model: nn.Module,
    train_dataset: CustomSequence,
    num_epochs: int = 10,
    val_dataset: CustomSequence = None,
    use_gpu: bool = True,
    data_parallel: bool = True,
    out_dir: str = ".",
    model_name: str = "model",
    verbose: bool = True,
    save_checkpoints: bool = True,
    custom_loss=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    if use_gpu:
        if device == "cpu":
            Warning("GPU not available, using CPU instead.")
        elif data_parallel:
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
                print("Using", torch.cuda.device_count(), "GPUs!")
            else:
                Warning("Data parallelism not available, using single GPU instead.")
        else:
            pass
    else:
        pass
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    train_losses = []
    if val_dataset is not None:
        val_losses = []

    # Train
    for epoch in range(num_epochs):
        if verbose:
            print("Epoch", epoch + 1, "of", num_epochs)

        model.train()
        train_loss = 0
        train_len = 0
        for i in range(len(train_dataset)):
            sample_generator = train_dataset[i]
            for X, y in sample_generator:
                # Erfan: I don't understand why this normalization is happening
                train_len += X.size(0)
                X, y = X.to(torch.float32), y.to(torch.float32)
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(X)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X.size(0)
        train_loss /= train_len
        # Erfan: Maybe delete the older checkpoints after saving the new one?
        # (So you wouldn't have terabytes of checkpoints just sitting there)
        if save_checkpoints:
            checkpoint_path = os.path.join(out_dir, f"{model_name}_epoch_{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                },
                checkpoint_path,
            )

        # Validation
        if val_dataset is not None:
            model.eval()
            val_loss = 0
            val_len = 0
            with torch.no_grad():
                for i in range(len(val_dataset)):
                    sample_generator = val_dataset[i]
                    for X, y in sample_generator:
                        X, y = X.to(torch.float32), y.to(torch.float32)
                        X, y = X.to(device), y.to(device)
                        val_len += X.size(0)
                        pred = model(X)
                        loss = criterion(pred, y)
                        val_loss += loss.item() * X.size(0)
            val_loss /= val_len

        train_losses.append(train_loss)
        np.save(os.path.join(out_dir, "train_losses.npy"), np.array(train_losses))
        if val_dataset is not None:
            val_losses.append(val_loss)
            np.save(os.path.join(out_dir, "val_losses.npy"), np.array(val_losses))

        if verbose:
            if val_dataset is not None:
                print(
                    f"Epoch {epoch + 1}: Train Loss={train_loss:.18f}, Val Loss={val_loss:.18f}"
                )
            else:
                print(f"Epoch {epoch + 1}: Train Loss={train_loss:.18f}")

    torch.save(model.state_dict(), os.path.join(out_dir, f"{model_name}.pth"))
    return model, train_losses, val_losses


def predict(
    model: nn.Module,
    model_param_path: str = None,
    test_dataset: CustomSequence = None,
    use_gpu: bool = True,
    data_parallel: bool = False,
    output_dir: str = ".",
    output_name: str = "all_preds.npy",
    verbose: bool = True,
):
    if model_param_path is not None:
        params = torch.load(model_param_path)
        if type(params) == dict and "model_state_dict" in params:
            params = params["model_state_dict"]
        try:
            model.load_state_dict(params)
        except:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(params)

    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    if use_gpu:
        if device == "cpu":
            Warning("GPU not available, using CPU instead.")
        if data_parallel:
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            else:
                Warning("Data parallelism not available, using single GPU instead.")
        else:
            try:
                model = model.module()
            except:
                pass
    model.to(device)

    model.eval()
    all_preds = []
    final_shape = None
    with torch.no_grad():
        for i in range(len(test_dataset)):
            if verbose:
                print(f"Predicting batch {i+1}/{len(test_dataset)}")
            sample_generator = test_dataset[i]
            for X, y in sample_generator:
                X, y = X.to(torch.float32), y.to(torch.float32)
                X, y = X.to(device), y.to(device)
                if final_shape is None:
                    final_shape = y.shape[1]
                pred = model(X)
                for _ in range(100):  # need to predict 100 times
                    pred = model(X)
                    X = X[:, 1:, :]  # pop first
                    X = torch.cat(
                        (X, torch.reshape(pred, (-1, 1, final_shape))), 1
                    )  # add to last
                all_preds.append(pred.squeeze().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    np.save(os.path.join(output_dir, f"{output_name}"), all_preds)


# Erfan: I think all of this "giving the data to the model" should be repackaged into
# one function and not be repeated everywhere.