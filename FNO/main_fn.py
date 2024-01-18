
from Utilz.loads import get_datasets

from argparse import Namespace
from FNO.model import NeuralOperator as FNO

from Utilz.main_fn import main_function


def main_FNO(
    args: Namespace,
):
    # Iniiate FNO model with explicit parameters
    model_dict = {
        "input_dim": 8264,
        "lifted_dim": args.FNO_lifted_dim,
        "output_dim": 8264,
        "num_layers": args.FNO_fourier_layers,
        "n_modes": args.FNO_modes,
        "is_slice": bool(args.is_slice),
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
