from typing import Callable
from GAN.training import gan_train
from Utilz.data import CustomSequence

from argparse import Namespace


def main_gan(
    args: Namespace,
    train_dataset: CustomSequence,
    val_dataset: CustomSequence,
    test_dataset: CustomSequence,
    custom_loss: Callable,
):
    gan_train(
        input_dim=8264,
        hidden_dim=128,
        output_dim=8264,
        num_epochs=args.num_epochs,
        train_dataset=train_dataset,
        lr=0.001,
    )
