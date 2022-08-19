from Ops import *
import torch.nn as nn
from Args import args as A
from torch.autograd import Variable as V
import torch
from Utils import *

T = torch.Tensor
M = nn.Module
class Cell(M):
    def __init__(self, type_, path_num) -> None:
        super().__init__()
        self.type_ = type_
        self.path_num = path_num
        self.node_num = A.node_num
        self.c_in, self.c_out = A.channles[self.type_]
        # (H ,W) 
        self.shape_in, self.shape_out = A.cell_shape[self.type_]
        self.cal_graph = self.init_graph()
        self.converge = nn.Conv2d(self.c_in*self.node_num, 1, kernel_size=1, stride=1, padding=0)
    
    def init_graph(self) -> nn.ModuleList:
        """初始化cell中的计算图
        """
        ops = nn.ModuleList()
        for _ in range(self.path_num):
            ops.append(MixOP(self.shape_in, self.shape_out, self.c_in, self.c_out, self.type_))
        return ops
    
    def reg_op(self, x) -> T:
        if self.type_ == 'e2n':
            x = torch.sum(x, dim=3,keepdim=True)
        if self.type_ == 'n2g':
            x = torch.sum(x, dim=2,keepdim=True)
        return x
    
    def forward(self, x, *arg, **kwargs) -> T:
        # !important 计算顺序: [in, out] <0,1> <0,2> <1,2> <0,3> <1,3> <2,3> ...
        #       +-----------------------------------------+  
        #       |                                         |  
        #       |                                         |  
        #    +----+         +---+   +----+                |  
        # +--|Node| ---P----+ADD+-->|Node|------+         |  
        # |  +----+         +--++   +----+      |         v  
        # |     |              |                |       +---+
        # |     P    +----+    P                +------>|cat|
        # |     +--->|Node|----+----------------+        +---+
        # |          +----+                               ^  
        # |                                               |  
        # +-----------------------------------------------+  
        # * 每个cell 的input不参与聚合
        # todo 每次前向传播都按照概率激活合法路径
        states = [x]
        p = kwargs['p'] # 路径概率
        path_idx = 0
        for _ in range(self.node_num):
            h_state = 0
            for state in states:
                h_ = self.cal_graph[path_idx](state)
                h_.mul_(p[path_idx])
                h_state += h_
                path_idx += 1
            states.append(h_state)
        out = torch.cat(states[1::], dim=1)
        out = self.reg_op(out)
        print(f"cell type:{self.type_},out size:{out.shape}")
        return out, _