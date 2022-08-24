import torch.nn as nn
import torch
from Args import args as A

M = nn.Module


def undefined(func):
    def with_logging(*args, **kwargs):
        print(f"The called function or objectve is undefined! Please check!")
        return func(*args, **kwargs)
    return with_logging

# OPS = {
#     'e2e': ['E2E','skip','none'],
#     'e2n': ['E2N','skip','none'],
#     'n2g': ['N2G','skip','none'],
# }

OPS = {
    'e2e': ['E2E'],
    'e2n': ['E2N'],
    'n2g': ['N2G'],
}

MetaOP = {
    'none': lambda shape_in, shape_out, c_in, c_out, type_: Zeros(shape_in, shape_out, c_in, c_out, type_),
    'skip': lambda shape_in, shape_out, c_in, c_out, type_: Skip(shape_in, shape_out, c_in, c_out, type_),
    'E2E' : lambda shape_in, shape_out, c_in, c_out, type_: E2E(shape_in, shape_out, c_in, c_out, type_),
    'E2N' : lambda shape_in, shape_out, c_in, c_out, type_: E2N(shape_in, shape_out, c_in, c_out, type_),
    'N2G' : lambda shape_in, shape_out, c_in, c_out, type_: N2G(shape_in, shape_out, c_in, c_out, type_),
}

def get_RDB(c_out):
    return nn.Sequential(
        nn.LeakyReLU(A.op_leak_relu, inplace=False),
        nn.BatchNorm2d(c_out, affine=True),
        nn.Dropout(A.drop_prob)
        )


class BaseOP(M):
    def __init__(self, shape_in, shape_out, c_in, c_out, type_) -> None:
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.type_ = type_
        self.reg_op_ = set_reg_op(shape_in, shape_out, c_in, c_out, type_)
        self.reg_op = nn.ModuleDict({
            'c': self.reg_op_['c'],
            'w': self.reg_op_['w'],
            'h': self.reg_op_['h']
        })
        del self.reg_op_
        self.RBD = get_RDB(c_out)

# 对输入和输出需要规范形状
# 由于在cell内部的node之间需要相互传递输入输出 为了能够正常运算 这里提前规定在node之间的数据流形状一致 为输出形状  

# RegOPs => 规范算子
# RegOPs_[C][W][H] => 针对通道、宽、高进行规范化
# RegOPs_*[Conv][..] => 使用的规范化操作是卷积还是复制

# C: B 1 W H <-> B 32 W H   e2e
# W: B C 90 90 <-> B C 1 90 e2n
# H: B C 1 90 <-> B C 1 1   n2g

class RegOPsCConv(M):
    def __init__(self, c_in, c_out, **kwargs) -> None:
        super().__init__()
        self.op = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, x):
        # print(f"conv reg before:{x.shape}")
        x = self.op(x)
        # print(f"conv reg after:{x.shape}")
        return x
    
class RegOPsCCopy(M):
    def __init__(self, c_in, c_out, **kwargs) -> None:
        super().__init__()
        self.c = [c_in, c_out]
        self.type_ = kwargs['type_']
    def forward(self, x):
        # print(f"conv reg before({self.type_}):{x.shape}")
        com = self.c[1]//self.c[0] + self.c[1]%self.c[0]
        # print(f"conv reg after:{x.shape}")
        return torch.cat([x]*com,1)

# W是针对的e2n的规范化
# e2n是接受90x90 输出90x1
# 所以规范化应该是扩充

class RegOPsWConv(M):
    @undefined
    def __init__(self,c_in, c_out, **kwargs) -> None:
        super().__init__()
        # todo
    @undefined
    def forward(self, x):
        # todo
        return x
    
class RegOPsWCopy(M):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.type_ = kwargs['type_']
    def forward(self, x):
        # print(f"w reg before({self.type_}):{x.shape}")
        x = torch.cat([x]*A.in_size[2], 2)
        # print(f"w reg after:{x.shape}")
        return x
    
# H是针对的n2g的规范化
# e2n是接受90x1 输出1x1
# 所以规范化应该是扩充

class RegOPsHConv(M):
    @undefined
    def __init__(self, *, c_in, c_out, **kwargs) -> None:
        super().__init__()
        # todo
    @undefined
    def forward(self, x):
        # todo
        return x
        
class RegOPsHCopy(M):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.type_ = kwargs['type_']
    def forward(self, x):
        # print(f"h reg before({self.type_}):{x.shape}")
        x = torch.cat([x]*A.in_size[2], 3)
        # print(f"h reg after:{x.shape}")
        return x

def set_reg_op(shape_in, shape_out, c_in, c_out, type_) -> dict:
    reg_op = {}
    
    if A.reg_op['c'] == 'Conv':
        reg_op['c'] = RegOPsCConv(c_in=c_in, c_out=c_out, type_=type_)
    elif A.reg_op['c'] == 'Copy':
        reg_op['c'] = RegOPsCCopy(c_in=c_in, c_out=c_out, type_=type_)
    else:
        raise ValueError('Not Find RegOPs')
        

    if A.reg_op['w'] == 'Conv':
        reg_op['w'] = RegOPsWConv(c_in=c_in, c_out=c_out, type_=type_)
    elif A.reg_op['w'] == 'Copy':
        reg_op['w'] = RegOPsWCopy(shape_in=shape_in, shape_out=shape_out, type_=type_)
    else:
        raise ValueError('Not Find RegOPs')

    if A.reg_op['h'] == 'Conv':
        reg_op['h'] = RegOPsHConv(c_in=c_in, c_out=c_out, type_=type_)
    elif A.reg_op['h'] == 'Copy':
        reg_op['h'] = RegOPsHCopy(shape_in=shape_in, shape_out=shape_out, type_=type_)
    else:
        raise ValueError('Not Find RegOPs')
    return reg_op

class Zeros(BaseOP):
    def __init__(self, shape_in, shape_out, c_in, c_out, type_) -> None:
        super().__init__(shape_in, shape_out, c_in, c_out, type_)

    def forward(self, x):
        x = x.mul(0.)
        if x.shape[1] != A.channles_constraint[self.type_]:
            x = self.reg_op['c'](x)
        if x.shape[3] != self.shape_out[1]:
            x = self.reg_op['h'](x)
        if x.shape[2] != self.shape_out[0]:
            x = self.reg_op['w'](x)
        # print(f"zero op in {self.type_} cell, out size:{x.shape}")
        return x

class Skip(BaseOP):
    def __init__(self, shape_in, shape_out, c_in, c_out, type_) -> None:
        super().__init__(shape_in, shape_out, c_in, c_out, type_)
    def forward(self, x):
        if x.shape[1] != A.channles_constraint[self.type_]:
            x = self.reg_op['c'](x)
        if x.shape[3] != self.shape_in[1]:
            x = self.reg_op['h'](x)
        if x.shape[2] != self.shape_in[0]:
            x = self.reg_op['w'](x)
        # print(f"skip op in {self.type_} cell, out size:{x.shape}")
        return x
    
class E2E(BaseOP):
    def __init__(self, shape_in, shape_out, c_in, c_out, type_) -> None:
        super().__init__(shape_in, shape_out, c_in, c_out, type_)
        self.op1 = nn.Sequential(
            nn.Conv2d(self.c_out, self.c_out, (1, A.in_size[2]), bias=True),
            self.RBD,
        )
        self.op2 = nn.Sequential(
            nn.Conv2d(self.c_out, self.c_out, (A.in_size[2], 1), bias=True),
            self.RBD,
        )
    def forward(self, x):
        # 输入的通道可能不满足要求
        if x.shape[1] != A.channles_constraint['e2e']:
            x = self.reg_op['c'](x)
        x1 = self.op1(x)
        x2 = self.op2(x)
        x = torch.cat([x1]*A.in_size[2], 3) + torch.cat([x2]*A.in_size[2], 2)
        # print(f"e2e op in {self.type_} cell, out size:{x.shape}")
        return x

class E2N(BaseOP):
    def __init__(self, shape_in, shape_out, c_in, c_out, type_) -> None:
        super().__init__(shape_in, shape_out, c_in, c_out, type_)
        self.op = nn.Sequential(
            nn.Conv2d(self.c_out, self.c_out, (1, A.in_size[2]), bias=True),
            self.RBD,
        )
    def forward(self, x):
        if x.shape[1] != A.channles_constraint['e2n']:
            x = self.reg_op['c'](x)
        if x.shape[3] != self.shape_in[1]:
            x = self.reg_op['h'](x)
        if x.shape[2] != self.shape_in[0]:
            x = self.reg_op['w'](x)
        x = self.op(x)
        # print(f"e2n op in {self.type_} cell, out size:{x.shape}")
        return x

class N2G(BaseOP):
    def __init__(self, shape_in, shape_out, c_in, c_out, type_) -> None:
        super().__init__(shape_in, shape_out, c_in, c_out, type_)
        self.op = nn.Sequential(
            nn.Conv2d(self.c_out, self.c_out, (A.in_size[2], 1), bias=True),
            self.RBD,
        )
    def forward(self, x):
        if x.shape[1] != A.channles_constraint['n2g']:
            x = self.reg_op['c'](x)
        if x.shape[2] != self.shape_in[0]:
            x = self.reg_op['w'](x)
        x = self.op(x)
        if x.shape[3] != self.shape_in[1]:
            x = self.reg_op['h'](x)
        # print(f"n2g op in {self.type_} cell, out size:{x.shape}")
        return x

