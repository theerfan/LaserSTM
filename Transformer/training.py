import torch.optim as optim
import torch
import torch.nn as nn

from Transformer.model import TimeSeriesTransformer

# Hyperparameters
BATCH_SIZE = 64
LR = 0.001
NUM_EPOCHS = 100

# Hyperparameters
D_MODEL = 512  # Embedding dimension
NHEAD = 8  # Number of attention heads
NUM_ENCODER_LAYERS = 6  # Number of encoder layers
NUM_DECODER_LAYERS = 6  # Number of decoder layers
DROPOUT = 0.1  # Dropout rate


# Assuming you're using a GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Move the model to device
model = TimeSeriesTransformer(
    D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DROPOUT
).to(device)

# Define the loss function and the optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
optimizer = optim.Adam(model.parameters(), lr=LR)

# Assuming you have a DataLoader `dataloader` that loads source and target sequences
# Each batch should return (src, tgt) where:
# src: Source sequence (input time series). 
# Shape: (S, N, E) where S is the source sequence length, N is the batch size, E is the feature number.
# tgt: Target sequence (shifted input time series). 
# Shape: (T, N, E) where T is the target sequence length.

# TODO: Replace this with your own DataLoader
dataloader = None

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for i, (src, tgt) in enumerate(dataloader, 0):
        src, tgt = src.to(device), tgt.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(
            src, tgt[:-1, :, :]
        )  # Exclude last item from tgt as it doesn't have a subsequent item to predict

        # Compute loss
        loss = criterion(
            output, tgt[1:, :, :]
        )  # Exclude first item from tgt as it was the start token and doesn't have a preceding item

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Log the progress
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")
