import torch.nn as nn
from Operations import OPS, MetaOP
from torch.autograd import Variable as V
import torch

T = torch.Tensor
M = nn.Module

class MixOP(M):
    def __init__(self, shape_in, shape_out, c_in, c_out, type_) -> None:
        super().__init__()
        self.type_ = type_
        self.ops = nn.ModuleList()
        for op in OPS[type_]:
            self.ops.append(
                MetaOP[op](shape_in, shape_out, c_in, c_out, type_)
            )
    def forward(self, x, *args, **kwargs) -> T:
        sum_ = []
        for op in self.ops:
            sum_.append(op(x))
        return sum(sum_)
        # return sum(op(x) for op in self.ops)