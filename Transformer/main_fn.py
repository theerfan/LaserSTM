# TODO: Implement main function for transformer model

from Transformer.model import TransformerModel
from Utilz.data import CustomSequence
from typing import Callable
import logging


def main_transformer(args: dict, train_dataset: CustomSequence, val_dataset: CustomSequence, test_dataset: CustomSequence, custom_loss: Callable):
    model = TransformerModel(
            n_features=8264,
            n_predict=8264,
            n_head=2,
            n_hidden=128,
            n_enc_layers=2,
            n_dec_layers=2,
            dropout=0.1,
        )