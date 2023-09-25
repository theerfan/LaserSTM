import numpy as np
import torch
import torch.nn as nn
from neuralop import H1Loss, LpLoss
from neuralop.models import FNO
from neuralop import Trainer
from LSTM.utils import CustomSequence

# Read x_new_0.npy from processed_data
x = np.load("processed_dataset/X_new_0.npy")
x = torch.tensor(x, dtype=torch.float32)

# x.shape = (100, 10, 8264)
# (timesteps, spatial points, features)

# Iniiate FNO model with explicit parameters
# n_modes: tuple of ints, number of modes in each dimension (if 1d data, then (n_modes,) is going to work)
model = FNO(
    n_modes=(16,), hidden_channels=64, in_channels=x.shape[1], out_channels=x.shape[1]
)

# Do one forward pass
x_passed = model(torch.tensor(x, dtype=torch.float32))

# Let's load all x_new_0 to x_new_90.npy files and put them in a data loader
# We will use this data loader to train the model

train_dataset = CustomSequence(
    "processed_dataset/",
    range(0, 90),
    file_batch_size=1,
    model_batch_size=512,
    test_mode=False,
)

val_dataset = CustomSequence(
    "processed_dataset/",
    [90],
    file_batch_size=1,
    model_batch_size=512,
    test_mode=False,
)

test_dataset = CustomSequence(
    "processed_dataset/",
    range(91, 99),
    file_batch_size=1,
    model_batch_size=512,
    test_mode=True,
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=1, shuffle=False
)

val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset, batch_size=1, shuffle=False
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=1, shuffle=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

optimizer = torch.optim.Adam(model.parameters(), lr=8e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

trainer = Trainer(
    model=model,
    n_epochs=20,
    device=device,
    wandb_log=False,
    log_test_interval=3,
    use_distributed=False,
    verbose=True,
)

trainer.train(
    train_loader=train_loader,
    test_loaders=[val_loader, test_loader],
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=H1Loss(),
    eval_losses=[H1Loss(), LpLoss(p=2)],
)
