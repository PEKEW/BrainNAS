from ctypes import Union
from torch.autograd import Variable as V
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn import Parameter as P
from Args import args as A
from copy import deepcopy
from Cells import Cell
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
M = nn.Module
T = torch.Tensor
P= torch.nn.Parameter

class HyperNet(M):
    def __init__(self,*args,**kwargs) -> None:
        super().__init__()
        self.cell_type = ('e2e','e2n','n2g')
        self.in_size = A.in_size
        self.out_size = A.out_size
        self.cell_num = A.cell_num
        self.node_num = A.node_num
        
        self.path_num = self.init_path_num()
        self.path_prob = self.init_path_prob()
        self.cal_graph = self.init_cal_graph()
        self._test_var = self.init_path_prob()
        
    def init_path_num(self) -> int:
        """生成cell中路径个数
        """
        # 四个节点+一个输入 每个节点都和前面的所有节点相连
        return sum([_ for _ in range(self.node_num+1)])
        
        
    def init_path_prob(self) -> V:
        """生成路径概率
        每个cell都对应一组路径 所以形状是 cell_num * path_num
        """
        return V(1e-3 * torch.randn(self.cell_num, self.path_num), requires_grad=True)
    
    def get_path_prob(self) -> list:
        return [self.path_prob]
    
    def get_path_prob_(self) -> Iterator[V]:
        for path in self._test_var:
            yield path
        
    
    def init_cal_graph(self) -> nn.ModuleList:
        """初始化计算图

        Returns:
            graph: input -> cells -> tail_stem
        """
        graph = nn.ModuleList()
        for cell_type in self.cell_type:
            graph.append(Cell(cell_type, self.path_num))
        feature_num = A.liner_in*A.node_num
        tail_stem = nn.Sequential(
            nn.Linear(feature_num, int(feature_num//2)),
            nn.LeakyReLU(int(feature_num//2),inplace=False),
            nn.Dropout(A.drop_prob),
            nn.Linear(int(feature_num//2), A.out_size),
            nn.Softmax(dim=1)
        )
        tail_stem.type_ = "classifiler"
        graph.append(tail_stem)
        return graph
    
    def forward(self, x, *args, **kwargs) -> T:

        for cell_num, sub_graph in enumerate(self.cal_graph):
            if sub_graph.type_ == "classifiler":
                x = sub_graph(x.view(-1, A.liner_in*A.node_num))
            else:
                p = F.softmax(self.path_prob[cell_num], dim=0)
                x,_ = sub_graph(x, p=p)
        return x
    
    