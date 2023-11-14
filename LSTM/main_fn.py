from LSTM.model import LSTMModel
from Utilz.loads import get_datasets
from Utilz.main_fn import main_function


def main_lstm(
    args: dict,
):
    model = LSTMModel(input_size=8264, lstm_hidden_size=2048)
    train_dataset, val_dataset, test_dataset = get_datasets(args)

    main_function(
        args,
        model,
        train_dataset,
        val_dataset,
        test_dataset,
    )
