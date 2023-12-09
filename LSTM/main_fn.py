from LSTM.model import LSTMModel
from Utilz.loads import get_datasets
from Utilz.main_fn import main_function


def main_lstm(
    args: dict,
):

    model_dict = {
        "input_size": 8264,
        "lstm_hidden_size": args.lstm_hidden_size,
        "linear_layer_size": args.lstm_linear_layer_size,
        "num_layers": args.lstm_num_layers,
        "LSTM_dropout": args.lstm_dropout,
        "fc_dropout": args.fc_dropout,
    }

    model = LSTMModel(
        **model_dict,
    )
    train_dataset, val_dataset, test_dataset = get_datasets(args)

    main_function(
        args,
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        model_dict=model_dict,
    )
