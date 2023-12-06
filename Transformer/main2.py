import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerDecoderModel(nn.Module):
    def __init__(
        self,
        seq_len,
        vocab_size,
        d_model,
        nhead,
        num_decoder_layers,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super(TransformerDecoderModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(seq_len, dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src, memory):
        src = self.embedding(src) + self.pos_encoder(
            torch.arange(0, src.size(1), device=src.device)
        )
        output = self.transformer_decoder(src, memory)
        output = self.out(output)
        return output


# Model parameters
seq_len = 8264
vocab_size = 10000  # assuming a vocabulary size of 10000, adjust as needed
d_model = 512  # size of the embeddings and transformer
nhead = 8
num_decoder_layers = 6

model = TransformerDecoderModel(seq_len, vocab_size, d_model, nhead, num_decoder_layers)


class SequenceDataset(Dataset):
    def __init__(self, data_dir, x_prefix, y_prefix, num_samples):
        self.data_dir = data_dir
        self.x_prefix = x_prefix
        self.y_prefix = y_prefix
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x_path = os.path.join(self.data_dir, f"{self.x_prefix}_new_{idx}.npy")
        y_path = os.path.join(self.data_dir, f"{self.x_prefix}_new_{idx}.npy")
        x = np.load(x_path)
        y = np.load(y_path)
        return torch.from_numpy(x).long(), torch.from_numpy(y).long()


# Assuming you have only one sample for this demonstration
data_dir = "/mnt/oneterra/SFG_reIm_version1_reduced"
dataset = SequenceDataset(data_dir, "X", "y", 1)
dataloader = DataLoader(dataset, batch_size=1)

# Assuming model is already defined and loaded
# model = TransformerDecoderModel(...)
# model.load_state_dict(torch.load(PATH))

# Run the model on the data
model.eval()
with torch.no_grad():
    for X, Y in dataloader:
        # You might need to add code here to handle the shape of X as per your model's requirements
        # Also, create a memory tensor if required by your model, or modify the model to not use it
        prediction = model(X, memory=None)  # Modify as per your model's architecture

        a = 12

        # Process the prediction as needed
        # ...
