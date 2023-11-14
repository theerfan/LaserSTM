from LSTM.model import LSTMModel
from Utilz.training import (
    predict,
    train_and_test,
    tune_and_train,
)
from Utilz.loads import get_datasets, get_custom_loss
from Utilz.data import (
    CustomSequence,
)
import logging
import torch.nn as nn


def main_function(
    args: dict,
    model: nn.Module,
    train_dataset: CustomSequence,
    val_dataset: CustomSequence,
    test_dataset: CustomSequence,
):
    custom_loss = get_custom_loss(args)

    if args.do_prediction == 1:
        log_str = f"Prediction only mode for model {args.model}"
        print(log_str)
        logging.info(log_str)
        predict(
            model,
            model_param_path=args.model_param_path,
            test_dataset=test_dataset,
            output_dir=args.output_dir,
            output_name="all_preds.npy",
            verbose=args.verbose,
        )
    else:
        # This assumes that `tune_train` and `train_model` have the same signature
        # (as in required arguments)
        if args.tune_train == 1:
            function_to_exec = tune_and_train
            print_str = f"Tune train mode for model {args.model}"
        else:
            function_to_exec = train_and_test
            print_str = f"Training mode for model {args.model}"

        print(print_str)
        logging.info(print_str)

        function_to_exec(
            model,
            args.model_save_name,
            args.num_epochs,
            custom_loss,
            args.epoch_save_interval,
            args.output_dir,
            train_dataset,
            val_dataset,
            test_dataset,
            args.verbose,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            analysis_file_idx=args.analysis_file,
            analysis_item_idx=args.analysis_example,
            model_param_path=args.model_param_path,
        )
