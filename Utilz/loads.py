from argparse import Namespace
from functools import partial
from typing import Callable
from Utilz.losses import (
    pearson_corr,
    weighted_MSE,
    pseudo_energy_loss,
    area_under_curve_loss,
    wrapped_MSE,
    wrapped_BCE,
    wMSE_and_energy,
    normalized_weighted_MSE,
)
from Utilz.data import CustomSequence


def get_custom_loss(args: Namespace) -> Callable:
    loss_dict = {
        "weighted_MSE": weighted_MSE,
        "pearson_corr": pearson_corr,
        "MSE": wrapped_MSE,
        "BCE": wrapped_BCE,
        "pseudo_energy": pseudo_energy_loss,
        "area_under_curve": area_under_curve_loss,
        "wMSE_and_energy": wMSE_and_energy,
        "normalized_weighted_MSE": normalized_weighted_MSE,
    }

    def custom_loss(y_real, y_pred, shg_weight=None, sfg_weight=None):
        return loss_dict[args.custom_loss](
            y_real,
            y_pred,
            shg_weight=shg_weight,
            sfg_weight=sfg_weight,
            reduction=args.loss_reduction,
        )

    # If we have default values for the weights, then we can do a partial function
    if args.shg_weight is not None and args.sfg_weight is not None:
        # Do a partial function application
        custom_loss = partial(
            custom_loss, shg_weight=args.shg_weight, sfg_weight=args.sfg_weight
        )
    else:
        pass

    return custom_loss


def get_datasets(
    args: Namespace,
):
    load_in_gpu = bool(args.load_in_gpu)
    train_dataset = CustomSequence(
        args.data_dir,
        range(0, 90),
        load_mode=args.train_load_mode,
        crystal_length=args.crystal_length,
        load_in_gpu=load_in_gpu,
    )

    val_dataset = CustomSequence(
        args.data_dir,
        [90],
        load_mode=args.val_load_mode,
        crystal_length=args.crystal_length,
        load_in_gpu=load_in_gpu,
    )

    test_dataset = CustomSequence(
        args.data_dir,
        range(90, 100),
        load_mode=args.test_load_mode,
        crystal_length=args.crystal_length,
        load_in_gpu=load_in_gpu,
    )

    return train_dataset, val_dataset, test_dataset
