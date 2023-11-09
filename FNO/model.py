from torch import nn
from neuralop.models import FNO

class FNO_wrapper(FNO):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, x):
        out = super().forward(x)
        return out[:, -1, :]