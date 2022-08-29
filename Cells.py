
from Ops import *
import torch.nn as nn
from Args import args as A
import torch
from Utils import *
import torch.nn.functional as F

T = torch.Tensor
M = nn.Module

class CellList(nn.ModuleList):
    def __init__(self, modules=None):
        super().__init__(modules)
    def __getitem__(self, idx: int):
        idx = -1 if idx >= len(self) else idx
        return super().__getitem__(idx)

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
        
    
    def init_graph(self) -> nn.ModuleList:
        """初始化cell中的计算图
        """
        ops = CellList()
        for _ in range(self.path_num):
            ops.append(MixOP(self.shape_in, self.shape_out, self.c_in, self.c_out, self.type_))
        return ops
    
    
    def forward(self, x, *arg, **kwargs) -> T:
        # !important 计算顺序: [in, out] <0,1><0,2><1,2><0,3><1,3><2,3> ..
        states = [x]
        p = kwargs['p'] # 路径概率 size = 1,10
        path_idx = 0
        state_dct_copy = self.cal_graph[path_idx].state_dict()
        for _ in range(self.node_num):
            h_state = 0
            p_ = p[path_idx:path_idx+len(states)+1]
            softed_p = F.softmax(p_, dim=0)
            # for i,state in enumerate(states):
            #     if path_idx != 0:
            #         self.cal_graph[path_idx].load_state_dict(state_dct_copy)
            #         with torch.no_grad():
            #             h_ = self.cal_graph[path_idx](state)
            #     else:
            #         h_ = self.cal_graph[path_idx](state)
            #     # h_.mul_(softed_p[i])
            #     h_state += h_
            #     path_idx += 1
            # states.append(h_state)
            
            for i,state in enumerate(states):
                h_ = self.cal_graph[path_idx](state)
                # h_.mul_(softed_p[i])
                h_state += h_
                path_idx += 1
            states.append(h_state)
                
        return torch.cat(states[1::], dim=1)