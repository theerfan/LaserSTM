from LSTM.model import LSTMModel, BlindTridentLSTM, TridentLSTM
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
        "has_fc_dropout": args.has_fc_dropout,
        "bidirectional": args.bidirectional,
        "layernorm": args.layernorm,
    }

    if args.model == "LSTM":
        model = LSTMModel(
            **model_dict,
        )
    elif args.model == "TridentLSTM":
        model = TridentLSTM(
            **model_dict,
        )
    elif args.model == "BlindTridentLSTM":
        model = BlindTridentLSTM(
            **model_dict,
        )
    else:
        raise NotImplementedError
    train_dataset, val_dataset, test_dataset = get_datasets(args)

    main_function(
        args,
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        model_dict=model_dict,
    )
