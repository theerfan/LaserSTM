from typing import Callable
from GAN.training import gan_train
from Utilz.data import CustomSequence


def main_gan(
    args: dict,
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
        train_set=train_dataset,
        lr=0.001,
    )
