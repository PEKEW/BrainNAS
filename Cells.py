
from Ops import *
import torch.nn as nn
from Args import args as A
import torch
from Utils import *
import torch.nn.functional as F

T = torch.Tensor
M = nn.Module

class CellList(nn.ModuleList):
    """对ModuleList重写,当__getitem__的索引超出最大长度时, 返回为最后一个元素
    """
    def __init__(self, modules=None):
        super().__init__(modules)
    def __getitem__(self, idx: int) -> M:
        idx = -1 if idx >= len(self) else idx
        return super().__getitem__(idx)




class Cell(M):
    """cell 是模型的第二级单元 有三种类型: e2e e2n 和 n2g
    cell之间的连接方式是链式的
    """
    def __init__(self, type_, path_num) -> None:
        """
        Args:
            type_ (str): cell类型: [e2e, e2n, n2g]
            path_num (int): cell中有多少连接数, 继承自上一级模型Net
        Attributes:
            type_ str: cell类型
            path_num: int cell中的连接数, 和node_num有关
            # ! 两个相邻的cell输入和输出数量之间相差node_num倍
            c_in int: 输入的通道数量
            c_out int: 输出的通道数量
            shape_in tuple: 输入的形状 (W,H)
            shape_out tuple: 输出的形状
            cal_graph M: cell中的基本计算图
        """
        super().__init__()
        self.type_ = type_
        self.path_num = path_num
        self.node_num = A.node_num
        self.c_in, self.c_out = A.channles[self.type_]
        self.shape_in, self.shape_out = A.cell_shape[self.type_]
        self.cal_graph = self.init_graph()
        
    
    def init_graph(self) -> nn.ModuleList:
        """初始化cell中的计算图
        每个路径都绑定一个计算图
        """
        ops = CellList()
        for _ in range(self.path_num):
            ops.append(MixOP(self.shape_in, self.shape_out, self.c_in, self.c_out, self.type_))
        return ops
    
    
    def forward(self, x, *arg, **kwargs) -> T:
        # cell forward
        #                                                  +---+                               
        #                                                  | p |                               
        #                                                  +---+                               
        #                                                    |                                 
        #                                  +-----------------+                                 
        #                                  |                 |                                 
        # +---------+                      v         +-------+                                 
        # |states[0]|------------------soft_p[0]-----+-------+-----------+                     
        # +---------+                                |       |           |                     
        #                                            |       |           v                     
        #                 +---------+                v       |         +---+        +---------+
        #                 |states[1]|----------soft_p[1]-----+-------->|ADD|--cat-->|states[3]|
        #                 +---------+                        |         +---+        +---------+
        #                                                    |           ^                     
        #                                                    |           |                     
        #                               +---------+          v           |                     
        #                               |states[2]|------soft_p[2]-------+                     
        #                               +---------+                                       
        # !important 计算顺序: [in, out] <0,1><0,2><1,2><0,3><1,3><2,3> ..
        states = [x]
        p = kwargs['p'] # 路径概率 size = 1,path_num
        path_idx = 0
        for _ in range(self.node_num):
            h_state = 0
            p_ = p[path_idx:path_idx+len(states)+1]
            softed_p = F.softmax(p_, dim=0)
            for i,state in enumerate(states):
                h_ = self.cal_graph[path_idx](state)
                h_.mul_(softed_p[i])
                h_state += h_
                path_idx += 1
            states.append(h_state)
        return torch.cat(states[1::], dim=1)
    
    def unused__forward(self, x, *args, **kwargs) -> T:
        """另一种forward 类似于rnn 是共享权重的 用于加快推理 缩小网络尺寸
        """
        states = [x]
        p = kwargs['p']
        path_idx = 0
        state_dct_copy = self.cal_graph[path_idx].state_dict()
        for _ in range(self.node_num):
            h_state = 0
            p_ = p[path_idx:path_idx+len(states)+1]
            softed_p = F.softmax(p_, dim=0)
            for i,state in enumerate(states):
                if path_idx != 0:
                    self.cal_graph[path_idx].load_state_dict(state_dct_copy)
                    with torch.no_grad():
                        h_ = self.cal_graph[path_idx](state)
                else:
                    h_ = self.cal_graph[path_idx](state)
                h_.mul_(softed_p[i])
                h_state += h_
                path_idx += 1
            states.append(h_state)  
        return torch.cat(states[1::], dim=1)