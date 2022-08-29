from torch.autograd import Variable as V
from Operations import MetaOP
import torch.nn.functional as F
import torch.nn as nn
import torch
from Args import args as A
from Cells import Cell
from typing import Iterator, OrderedDict
M = nn.Module
T = torch.Tensor
P = torch.nn.Parameter

class BrainNetCNN(M):
    def __init__(self) -> None:
        super().__init__()
        self.cal_garph = self.init_graph()
        self.tail = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(64,inplace=False),
            nn.Linear(64, 32),
            nn.LeakyReLU(A.out_size,inplace=False),
            nn.Linear(32,16),
            nn.LeakyReLU(A.out_size,inplace=False),
            nn.Linear(16, A.out_size),
            nn.LeakyReLU(A.out_size,inplace=False),
            nn.Softmax(dim=1)
        )
    def init_graph(self):
        keys = ('E2E1','E2E2','E2N','N2G')
        shapes = {
            'E2E': [(90,90), (90,90)],
            'E2N': [(90,90),(90,1)],
            'N2G': [(90,1), (1,1)]
        }
        channels = (c for c in [1,32,32,32,32,64,64,128])
        dct = OrderedDict()
        for key in keys:
            k = key[0:3]
            dct[key] = MetaOP[k](shapes[k][0],shapes[k][1], next(channels), next(channels))
        return nn.Sequential(dct)
    def forward(self,x):
        x = self.cal_garph(x)
        x = self.tail(x.view(A.in_size[0],-1))
        return x

class SupHead(M):
    def __init__(self, feature_num) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_num, 64),
            nn.LeakyReLU(A.op_leak_relu),
            nn.Linear(64, 32),
            nn.LeakyReLU(A.op_leak_relu),
            nn.Linear(32,16),
            nn.LeakyReLU(A.op_leak_relu),
            nn.Linear(16, A.out_size),
            nn.LeakyReLU(A.op_leak_relu),
        )
    def forward(self, x):
        return self.classifier(x)
            
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
        self.sup_head = self.init_sup_head() if A.need_help else None
        
    def init_path_num(self) -> int:
        """生成cell中路径个数
        """
        # 四个节点+一个输入 每个节点都和前面的所有节点相连
        return sum([_ for _ in range(self.node_num+1)])
        
    def init_path_prob(self) -> V:
        """生成路径概率
        每个cell都对应一组路径 所以形状是 cell_num * path_num
        """
        # return V(torch.randn(self.cell_num, self.path_num), requires_grad=True)
        # return V(torch.ones(self.cell_num, self.path_num), requires_grad=True)
        return V(torch.randn(self.cell_num, self.path_num), requires_grad=False)
    
    def get_path_prob(self) -> list:
        return [self.path_prob]
    
    def get_path_prob_(self) -> Iterator[V]:
        for path in self._test_var:
            yield path
    
    def init_sup_head(self):
        sup_heads = nn.ModuleDict()
        for cell_type in self.cell_type:
            sup_heads[cell_type] = SupHead(
                #* feature num = C*H*W
                A.channles[cell_type][1]*self.node_num*A.shape_constraint[cell_type][1][1]*A.shape_constraint[cell_type][1][2]
                )
            
        return sup_heads
    def init_cal_graph(self) -> nn.ModuleList:
        """初始化计算图

        Returns:
            graph: input -> cells -> tail_stem
        """
        self.sup_head = self.init_sup_head()
        graph = nn.ModuleDict()
        for cell_type in self.cell_type:
            graph[cell_type] = Cell(cell_type, self.path_num)
            
        feature_num = A.liner_in*A.node_num
        tail_stem = nn.Sequential(
            nn.Linear(feature_num, 64),
            nn.LeakyReLU(A.op_leak_relu),
            nn.Linear(64, 32),
            nn.LeakyReLU(A.op_leak_relu),
            nn.Linear(32,16),
            nn.LeakyReLU(A.op_leak_relu),
            nn.Linear(16, A.out_size),
            nn.LeakyReLU(A.op_leak_relu),
        )
        tail_stem.type_ = "classifiler"
        graph[tail_stem.type_] = tail_stem
        return graph
    
    def forward(self, x, *args, **kwargs) -> T:
        for cell_num, (type_, sub_graph) in enumerate(self.cal_graph.items()):
            if type_ == "classifiler":
                x = sub_graph(x.view(A.in_size[0], -1))
            else:
                p = F.softmax(self.path_prob[cell_num], dim=0)
                p = self.path_prob[cell_num]
                x = sub_graph(x, p=p)
        return x, None
