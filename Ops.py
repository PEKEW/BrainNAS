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
        # self.ops = nn.ModuleList()
        self.ops = nn.ModuleDict()
        for op in OPS[type_]:
            self.ops[op] = MetaOP[op](shape_in, shape_out, c_in, c_out, type_)
            # self.ops.append(
            #     MetaOP[op](shape_in, shape_out, c_in, c_out, type_)
            # )
    def forward(self, x, *args, **kwargs) -> T:
        return sum(op(x) for op in self.ops.values())