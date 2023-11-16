
from Utilz.loads import get_datasets

from argparse import Namespace
from neuralop.models import FNO

from Utilz.main_fn import main_function


def main_FNO(
    args: Namespace,
):
    # Iniiate FNO model with explicit parameters
    # n_modes: tuple of ints, number of modes in each dimension (if 1d data, then (n_modes,) is going to work)
    
    if not args.is_slice:
        model = FNO(n_modes=(16,), hidden_channels=64, in_channels=1, out_channels=1)
        train_dataset, val_dataset, test_dataset = get_datasets(args, True, True, True)
    else:
        # 10 is the number of timesteps
        model = FNO(n_modes=(16,), hidden_channels=64, in_channels=10, out_channels=10)
        train_dataset, val_dataset, test_dataset = get_datasets(
            args, False, False, True
        )
    
    model_dict = {
        "n_modes": (16,),
        "hidden_channels": model.hidden_channels,
        "in_channels": model.in_channels,
        "out_channels": model.out_channels,
    }
    
    main_function(
        args,
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        model_dict=model_dict,
    )
