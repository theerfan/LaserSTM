from torch import nn
from neuralop.models import FNO

# The FNO wrapper is the class to use if we want to pass it
# 10 timesteps at once
class FNO_slice(FNO):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, x):
        out = super().forward(x)
        return out[:, -1, :]