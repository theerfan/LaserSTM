from argparse import Namespace
from functools import partial
from typing import Callable
from Utilz.data import (
    CustomSequence,
    pearson_corr,
    weighted_MSE,
    pseudo_energy_loss,
    area_under_curve_loss,
    wrapped_MSE,
    wrapped_BCE,
)


def get_custom_loss(args: Namespace) -> Callable:
    loss_dict = {
        "weighted_MSE": weighted_MSE,
        "pearson_corr": pearson_corr,
        "MSE": wrapped_MSE,
        "BCE": wrapped_BCE,
        "pseudo_energy": pseudo_energy_loss,
        "area_under_curve": area_under_curve_loss,
    }

    def custom_loss(y_real, y_pred, shg_weight=None, sfg_weight=None):
        return loss_dict[args.custom_loss](
            y_real, y_pred, shg_weight=shg_weight, sfg_weight=sfg_weight
        )

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
    train_test_mode: bool = False,
    val_test_mode: bool = False,
    test_test_mode: bool = True,
):
    train_dataset = CustomSequence(
        args.data_dir,
        range(0, 90),
        test_mode=train_test_mode,
        crystal_length=args.crystal_length,
    )

    val_dataset = CustomSequence(
        args.data_dir, [90], test_mode=val_test_mode, crystal_length=args.crystal_length
    )

    test_dataset = CustomSequence(
        args.data_dir,
        range(91, 100),
        test_mode=test_test_mode,
        crystal_length=args.crystal_length,
    )

    return train_dataset, val_dataset, test_dataset
