
from Utilz.loads import get_datasets

from argparse import Namespace
from neuralop.models import FNO


from Utilz.main_fn import main_function


def main_FNO(
    args: Namespace,
):
    # Iniiate FNO model with explicit parameters
    # n_modes: tuple of ints, number of modes in each dimension (if 1d data, then (n_modes,) is going to work)
    model = FNO(n_modes=(16,), hidden_channels=64, in_channels=10, out_channels=10)
    
    if args.is_slice:
        train_dataset, val_dataset, test_dataset = get_datasets(args, True, True, True)
    else:
        train_dataset, val_dataset, test_dataset = get_datasets(
            args, False, False, True
        )
    
    main_function(
        args,
        model,
        train_dataset,
        val_dataset,
        test_dataset,
    )
