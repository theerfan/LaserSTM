from Utilz.training import (
    predict,
    train_and_test,
    tune_and_train,
)
from Utilz.loads import get_custom_loss
from Utilz.data import (
    CustomSequence,
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

    if args.do_analysis == 1:
        log_str = f"Prediction only mode for model {args.model}"
        print(log_str)
        logging.info(log_str)
        model_save_name = os.path.basename(args.model_param_path).split(".")[0]
        do_analysis(
            args.output_dir,
            args.data_dir,
            model_save_name,
            file_idx=args.analysis_file,
            all_preds_idx=args.all_preds_idx,
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

        testset_starting_point = test_dataset.file_indexes[0]

        do_analysis(
            args.output_dir,
            args.data_dir,
            model_save_name,
            file_idx=args.analysis_file,
            all_preds_idx=testset_starting_point - args.analysis_file,
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
        )
