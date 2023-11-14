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

def get_custom_loss(args) -> Callable:
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


def get_datasets(args):
    train_dataset = CustomSequence(
        args.data_dir,
        range(0, 90),
    )

    val_dataset = CustomSequence(args.data_dir, [90])

    test_dataset = CustomSequence(
        args.data_dir,
        range(91, 100),
        load_in_gpu=False,
        # test_mode=True,
    )

    return train_dataset, val_dataset, test_dataset
