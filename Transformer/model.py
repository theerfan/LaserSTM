import torch
import torch.nn as nn


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self, d_model, nhead, num_encoder_layers, num_decoder_layers, dropout=0.5
    ):
        super(TimeSeriesTransformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
        )
        self.linear = nn.Linear(
            d_model, 1
        )  # 1 for single step prediction. Adjust for multistep prediction.

    def forward(self, src, tgt):
        # src: Source sequence (input time series).
        # Shape: (S, N, E) where S is the source sequence length, N is the batch size, E is the feature number.

        # tgt: Target sequence (shifted input time series).
        # Shape: (T, N, E) where T is the target sequence length.

        transformer_output = self.transformer(src, tgt)
        output = self.linear(transformer_output)
        return output


# Hyperparameters
D_MODEL = 512  # Embedding dimension
NHEAD = 8  # Number of attention heads
NUM_ENCODER_LAYERS = 6  # Number of encoder layers
NUM_DECODER_LAYERS = 6  # Number of decoder layers
DROPOUT = 0.1  # Dropout rate

model = TimeSeriesTransformer(
    D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DROPOUT
)
print(model)
