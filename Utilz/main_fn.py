from Utilz.training import predict, train_and_test, tune_and_train, funky_predict
from Utilz.loads import get_custom_loss
from Utilz.data import (
    CustomSequence,
    X_Dataset,
    Y_Dataset
)
import logging
import torch.nn as nn
import os

from argparse import Namespace

from Analysis.analyze_reim import do_analysis


def main_function(
    args: Namespace,
    model: nn.Module,
    train_dataset: CustomSequence,
    val_dataset: CustomSequence,
    test_dataset: CustomSequence,
    model_dict: dict = None,
):
    custom_loss = get_custom_loss(args)

    if args.do_funky == 1:
        log_str = f"Funky prediction mode for model {args.model}"
        print(log_str)
        logging.info(log_str)
        model_save_name = os.path.basename(args.model_param_path).split(".")[0]

        x_test_set = X_Dataset(args.data_dir, file_indexes=range(91, 100))
        y_test_set = Y_Dataset(args.data_dir, file_indexes=range(91, 100))
        
        funky_predict(
            model,
            model_param_path=args.model_param_path,
            x_dataset=x_test_set,
            y_dataset=y_test_set,
            output_dir=args.output_dir,
            x_batch_size=1,
            y_batch_size=100,
            model_save_name=model_save_name,
            verbose=args.verbose,
        )

    elif args.do_analysis == 1:
        log_str = f"Analysis only mode for model {args.model}"
        print(log_str)
        logging.info(log_str)
        model_save_name = os.path.basename(args.model_param_path).split(".")[0]
        do_analysis(
            args.output_dir,
            args.data_dir,
            model_save_name,
            file_idx=args.analysis_file,
            item_idx=args.analysis_example,
        )

    elif args.do_prediction == 1:
        log_str = f"Prediction only mode for model {args.model}"
        print(log_str)
        logging.info(log_str)
        model_save_name = os.path.basename(args.model_param_path).split(".")[0]
        predict(
            model,
            model_param_path=args.model_param_path,
            test_dataset=test_dataset,
            output_dir=args.output_dir,
            model_save_name=model_save_name,
            verbose=args.verbose,
        )

        do_analysis(
            args.output_dir,
            args.data_dir,
            model_save_name,
            file_idx=args.analysis_file,
            item_idx=args.analysis_example,
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
            model_dict=model_dict,
            learning_rate=args.lr,
            shuffle=args.shuffle,
        )
