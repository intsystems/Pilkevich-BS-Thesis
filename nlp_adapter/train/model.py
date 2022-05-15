from torch import nn
from torch import Tensor


class LinearAdapter(nn.Module):

    def __init__(self, in_size: int, out_size: int):
        super(LinearAdapter, self).__init__()
        self._in_size = in_size
        self._out_size = out_size

        self._linear_layer = nn.Linear(in_features=self._in_size, out_features=self._out_size)

    def forward(self, x: Tensor):
        out = self._linear_layer(x)
        return out
