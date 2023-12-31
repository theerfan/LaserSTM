import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def add_prefix(lst: list, prefix="X_new"):
    """
    Add prefix to list of file names
    @param lst: list of file names
    @param prefix: prefix to add
    @return: list of file names with prefix
    """
    return [prefix + "_" + str(i) + ".npy" for i in lst]


def CustomSequenceTiming(data_dir: str, file_idx: list, prefix="X_new"):
    Xnames = add_prefix(file_idx, prefix)
    X = None
    for x in Xnames:
        if X is None:
            X = np.load(os.path.join(data_dir, x))[::100]
        else:
            X = np.concatenate((X, np.load(os.path.join(data_dir, x))[::100]))
    return X


def time_prediction(
    model: nn.Module,
    model_param_path: str = None,
    test_dataset=None,
    batch_size: int = 200,
    verbose: bool = True,
):
    # Erfan: Check this re-write to make sure it works
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_param_path is not None:
        params = torch.load(model_param_path, map_location=device)
        if isinstance(params, dict) and "model_state_dict" in params:
            params = params["model_state_dict"]
        try:
            model.load_state_dict(params)
        except:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(params)

    if device == "gpu":
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:
        model = model.to(device)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model.to(device)
    model.eval()
    all_preds = None
    start_time = time.time()
    final_shape = None
    # i = 0
    with torch.no_grad():
        for j, X_batch in enumerate(test_dataloader):
            X_batch = X_batch.to(torch.float32)
            # if verbose:
            #     print(f"Predicting batch {i+1}/{len(test_dataloader)}")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X_batch = X_batch.to(device)

            if final_shape is None:
                final_shape = X_batch.shape[-1]

            for _ in range(100):  # need to predict 100 times
                pred = model(X_batch)
                X_batch = X_batch[:, 1:, :]  # pop first

                # add to last
                X_batch = torch.cat(
                    (X_batch, torch.reshape(pred, (-1, 1, final_shape))), 1
                )

            # Keep all_preds on GPU instead of sending it back to CPU at "each" iteration
            # Erfan TODO: Best if we know the value of *pred.squeeze().shape beforehand
            if all_preds is None:
                all_preds = torch.zeros(
                    (len(test_dataset), *pred.squeeze().shape), device=device
                )
            else:
                pass
            all_preds[j] = pred.squeeze()

    end_time = time.time()
    # And then we do the concatenation here and send it back to CPU
    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    return all_preds, end_time - start_time
