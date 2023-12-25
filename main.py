import logging

import argparse

from LSTM.main_fn import main_lstm
from GAN.main_fn import main_gan
from Transformer.main_fn import main_transformer
from FNO.main_fn import main_FNO


logging.basicConfig(
    filename="application_log.log", level=logging.INFO, format="%(message)s"
)


def get_cmd_args():
    parser = argparse.ArgumentParser(description="Train and test the model.")
    parser.add_argument(
        "--model", type=str, required=True, help="Model to use for training."
    )
    parser.add_argument(
        "--model_save_name",
        type=str,
        default="LSTM_model_latest",
        help="Model to use for training.",
    )

    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to the data directory."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="Number of epochs for training."
    )
    parser.add_argument(
        "--custom_loss", type=str, default="MSE", help="Custom loss function name."
    )

    # Add arguments for the shg and sfg weight losses
    parser.add_argument(
        "--shg_weight", type=float, default=None, help="Weight for the SHG loss."
    )

    parser.add_argument(
        "--sfg_weight", type=float, default=None, help="Weight for the SFG loss."
    )

    parser.add_argument(
        "--epoch_save_interval", type=int, default=1, help="Epoch save interval."
    )

    parser.add_argument(
        "--tune_train",
        type=int,
        default=0,
        help="Whether to do hyperparameter tuning or not.",
    )

    parser.add_argument("--output_dir", type=str, default=".", help="Output directory.")

    parser.add_argument(
        "--do_prediction",
        type=int,
        default=0,
        help="Whether to do prediction or not.",
    )

    parser.add_argument(
        "--do_analysis",
        type=int,
        default=0,
        help="Whether to do prediction or not.",
    )

    parser.add_argument(
        "--do_funky",
        type=int,
        default=0,
        help="Whether to do funky prediction or not.",
    )

    parser.add_argument(
        "--model_param_path",
        type=str,
        default=None,
        help="Path to the model parameters.",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Whether to print the progress or not.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=7_500,
        help="How large should the batch be to maximize GPU memory util",
    )

    # This would use [testset_last_file - 91] file, beca
    parser.add_argument(
        "--analysis_file",
        type=int,
        default=91,
        help="The file to use for final analysis.",
    )

    parser.add_argument(
        "--analysis_example",
        type=int,
        default=15,
        help="The example of the file to use for analysis",
    )

    parser.add_argument(
        "--crystal_length",
        type=int,
        default=100,
        help="The assumed length for our crystal",
    )

    parser.add_argument(
        "--is_slice",
        type=bool,
        default=True,
        help="Are we modeling a slice of the crystal or the whole crystal?",
    )

    parser.add_argument(
        "--lstm_hidden_size",
        type=int,
        default=1024,
        help="Hidden size of the LSTM model",
    )

    parser.add_argument(
        "--lstm_num_layers",
        type=int,
        default=1,
        help="Number of layers of the LSTM model",
    )

    parser.add_argument(
        "--lstm_linear_layer_size",
        type=int,
        default=4096,
        help="Number of layers of the LSTM model",
    )

    parser.add_argument(
        "--loss_reduction",
        type=str,
        default="mean",
        help="Reduction type for the loss function.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for the model.",
    )

    parser.add_argument(
        "--lstm_dropout",
        type=float,
        default=0.3,
        help="Dropout rate for the LSTM model.",
    )

    parser.add_argument(
        "--fc_dropout",
        type=float,
        default=0.3,
        help="Dropout rate for the fully connected layers.",
    )

    # 0 for training mode, 1 for test mode
    # and 2 for "step-wise analysis" mode

    parser.add_argument(
        "--train_load_mode",
        type=int,
        default=0,
        help="Load the train dataset in test mode.",
    )

    parser.add_argument(
        "--val_load_mode",
        type=int,
        default=0,
        help="Load the val dataset in test mode.",
    )

    parser.add_argument(
        "--test_load_mode",
        type=int,
        default=1,
        help="Load the test dataset in test mode.",
    )

    parser.add_argument(
        "--shuffle",
        type=int,
        default=1,
        help="Whether to shuffle the training dataset or not.",
    )

    parser.add_argument(
        "--has_fc_dropout",
        type=int,
        default=1,
        help="Whether to have the dropout layers in the post-LSTM linear layers.",
    )

    return parser.parse_args()


# Currently using v2 (reIm) data
if __name__ == "__main__":
    # Get the args from command line
    args = get_cmd_args()

    if args.model == "LSTM" or args.model == "TridentLSTM":
        main_lstm(args)
    elif args.model == "Transformer":
        main_transformer(args)
    elif args.model == "GAN":
        main_gan(args)
    elif args.model == "FNO":
        main_FNO(args)
    else:
        raise ValueError("Model not supported.")
