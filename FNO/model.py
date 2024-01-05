from torch import nn
from neuralop.models import FNO


class FNO_wrapper(FNO):
    def __init__(self, is_slice: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.is_slice = is_slice

        # print the kwargs
        print("FNO_wrapper kwargs:")
        for k, v in kwargs.items():
            print(f"{k}: {v}")
        print(f"is_slice: {is_slice}")

    def forward(self, x):
        # if x has a second dimension, take the last element
        # this is due to the way the data is stored
        if len(x.shape) == 3 and not self.is_slice:
            x = x[:, -1, :]
        out = super().forward(x)
        return out
