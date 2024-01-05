
from Utilz.loads import get_datasets

from argparse import Namespace
from FNO.model import FNO_wrapper as FNO

from Utilz.main_fn import main_function


def main_FNO(
    args: Namespace,
):
    # Iniiate FNO model with explicit parameters
    # n_modes: tuple of ints, number of modes in each dimension (if 1d data, then (n_modes,) is going to work)
    
    n_data_channels = 1 if not args.is_slice else 10

    model_dict = {
        "n_modes": (args.FNO_modes,),
        "hidden_channels": args.FNO_hidden_channels,
        "in_channels": n_data_channels,
        "out_channels": n_data_channels,
        "mlp_dropout": args.fc_dropout if args.has_fc_dropout else 0,
        "use_mlp": bool(args.FNO_use_mlp),
        "n_layers": args.FNO_fourier_layers,
        "is_slice": args.is_slice,
    }

    train_dataset, val_dataset, test_dataset = get_datasets(args)
    
    model = FNO(**model_dict)
    
    main_function(
        args,
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        model_dict=model_dict,
    )
