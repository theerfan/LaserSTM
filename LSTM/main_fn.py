from LSTM.model import LSTMModel
from LSTM.training import (
    predict,
    test_train_lstm,
    tune_train_lstm,
)
from LSTM.utils import CustomSequence
from typing import Callable
import logging


def main_lstm(
    args: dict,
    train_dataset: CustomSequence,
    val_dataset: CustomSequence,
    test_dataset: CustomSequence,
    custom_loss: Callable,
):
    model = LSTMModel(input_size=8264)
    if args.do_prediction == 1:
        log_str = f"Prediction only mode for model {args.model}"
        print(log_str)
        logging.info(log_str)
        predict(
            model,
            model_param_path=args.model_param_path,
            test_dataset=test_dataset,
            data_parallel=False,
            output_dir=args.output_dir,
            output_name="all_preds.npy",
            verbose=1,
        )
    else:
        # This assumes that `tune_train` and `train_model` have the same signature
        # (as in required arguments)
        if args.tune_train == 1:
            function_to_exec = tune_train_lstm
            print_str = f"Tune train mode for model {args.model}"
        else:
            function_to_exec = test_train_lstm
            print_str = f"Training mode for model {args.model}"

        print(print_str)
        logging.info(print_str)

        function_to_exec(
            model,
            args.num_epochs,
            custom_loss,
            args.epoch_save_interval,
            args.output_dir,
            train_dataset,
            val_dataset,
            test_dataset,
            args.verbose,
            data_dir=args.data_dir,
        )