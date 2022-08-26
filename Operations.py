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
    'e2e': ['skip'],
    'e2n': ['skip'],
    'n2g': ['skip'],
}

MetaOP = {
    'Test': lambda shape_in, shape_out, c_in, c_out, type_: Test(shape_in, shape_out, c_in, c_out, type_),
    'none': lambda shape_in, shape_out, c_in, c_out, type_: Zeros(shape_in, shape_out, c_in, c_out, type_),
    'skip': lambda shape_in, shape_out, c_in, c_out, type_: Skip(shape_in, shape_out, c_in, c_out, type_),
    'E2E' : lambda shape_in, shape_out, c_in, c_out, type_=None: E2E(shape_in, shape_out, c_in, c_out, type_),
    'E2N' : lambda shape_in, shape_out, c_in, c_out, type_=None: E2N(shape_in, shape_out, c_in, c_out, type_),
    'N2G' : lambda shape_in, shape_out, c_in, c_out, type_=None: N2G(shape_in, shape_out, c_in, c_out, type_),
}

def get_RDB(c_out):
    return nn.Sequential(
        nn.LeakyReLU(A.op_leak_relu),
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
        self.reg_op_in = set_reg_op(c_in, c_out, type_, 'in')
        self.reg_op_out = set_reg_op(c_in, c_out, type_,)
        # self.RBD = nn.Sequential(
        # nn.LeakyReLU(A.op_leak_relu),
        # nn.BatchNorm2d(c_out, affine=True),
        # nn.Dropout(A.drop_prob)
        # )
    
    def set_reg_op(self):
        if self.type_ == 'e2e':
            self.reg_in = self.reg_op_in['c']
            self.reg_out = self.reg_op_out['c']
        elif self.type_ == 'e2n':
            self.reg_in = self.reg_op_in['w']
            self.reg_out =  self.reg_op_out['w']
        else:
            self.reg_in = self.reg_op_in['h']
            self.reg_out = self.reg_op_out['h']
        del self.reg_op_in, self.reg_op_out
    

    # def forward(self, x):
    #     x = self.reg_in(x)
    #     x = x.mul(0.)
    #     x = self.RBD(x)
    #     x = self.reg_out(x)
    #     return x

# 对输入和输出需要规范形状
# 由于在cell内部的node之间需要相互传递输入输出 为了能够正常运算 这里提前规定在node之间的数据流形状一致 为输出形状  

# RegOPs => 规范算子
# RegOPs_[C][W][H] => 针对通道、宽、高进行规范化

# C: B 1 W H <-> B 32 W H   e2e
# W: B C 90 90 <-> B C 1 90 e2n
# H: B C 1 90 <-> B C 1 1   n2g
    
class RegOPsCCopyIn(M):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.c = [kwargs['c_in'], kwargs['c_out']]
        self.type_ = kwargs['type_']
    def forward(self, x):
        if x.shape[1] < A.shape_constraint[self.type_][0][0]:
            x = torch.cat([x]*(self.c[1]//self.c[0] + self.c[1]%self.c[0]),1)
        return x

class RegOPsWCopyIn(M):
    def __init__(self, **kwargs) -> None:
        super().__init__()
    def forward(self, x):
        if x.shape[2] == 1:
            x = torch.cat([x]*A.in_size[3], 2)
        return x

class RegOPsHCopyIn(M):
    def __init__(self, **kwargs) -> None:
        super().__init__()
    def forward(self, x):
        if x.shape[3] == 1:
            x = torch.cat([x]*A.in_size[2], 3)
        return x

class RegOPsCCopyOut(M):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.c = [kwargs['c_in'], kwargs['c_out']]
        self.type_ = kwargs['type_']
    def forward(self, x):
        # if x.shape[1] > A.shape_constraint[self.type_][1][0]:
        #     x = torch.cat([x]*(self.c[1]//self.c[0] + self.c[1]%self.c[0]),1)
        return x

class RegOPsWCopyOut(M):
    def __init__(self, **kwargs) -> None:
        super().__init__()
    def forward(self, x):
        if x.shape[2] != 1:
            x = torch.sum(x, 2, keepdim=True)
        return x

class RegOPsHCopyOut(M):
    def __init__(self, **kwargs) -> None:
        super().__init__()
    def forward(self, x):
        if x.shape[3] != 1:
            x = torch.sum(x, 3, keepdim=True)
        return x
    
def set_reg_op(c_in, c_out, type_, mode=None) -> dict:
    if mode == 'in':
        return  nn.ModuleDict({
            'c': RegOPsCCopyIn(c_in=c_in, c_out=c_out, type_=type_),
            'w': RegOPsWCopyIn(),
            'h': RegOPsHCopyIn()
        })
    else:
        return nn.ModuleDict({
            'c': RegOPsCCopyOut(c_in=c_in, c_out=c_out, type_=type_),
            'w': RegOPsWCopyOut(),
            'h': RegOPsHCopyOut()
        })

class Zeros(BaseOP):
    def __init__(self, shape_in, shape_out, c_in, c_out, type_) -> None:
        super().__init__(shape_in, shape_out, c_in, c_out, type_)
        self.set_reg_op()
        self.op = lambda x:x.mul(0.)

class Skip(BaseOP):
    def __init__(self, shape_in, shape_out, c_in, c_out, type_) -> None:
        super().__init__(shape_in, shape_out, c_in, c_out, type_)
        self.set_reg_op()
        self.op = lambda x:x
        self.__RBD = nn.Sequential(nn.LeakyReLU(A.op_leak_relu),
        nn.BatchNorm2d(c_out, affine=True),
        nn.Dropout(A.drop_prob))
        self.__reg_in = self.reg_in
        self.__reg_out = self.reg_out
    def forward(self, x):
        x = self.__reg_in(x)
        x = self.__RBD(x)
        x = self.__reg_out(x)
        return x

class Test(M):
    def __init__(self, shape_in, shape_out, c_in, c_out, type_) -> None:
        super().__init__()
        self.type_ = type_
        self.op = nn.Sequential(nn.LeakyReLU(A.op_leak_relu),
        nn.BatchNorm2d(1, affine=True),
        nn.Dropout(A.drop_prob))

    def forward(self, x,p=0):
        x = torch.sum(x,dim=1,keepdims=True)
        return self.op(x)
    
class E2E(BaseOP):
    def __init__(self, shape_in, shape_out, c_in, c_out, type_) -> None:
        super().__init__(shape_in, shape_out, c_in, c_out, type_)
        self.op1 = nn.Sequential(
            nn.Conv2d(self.c_out, self.c_out, (1, A.in_size[2]), bias=True),
            self.RBD,)
        self.op2 = nn.Sequential(
            nn.Conv2d(self.c_out, self.c_out, (A.in_size[2], 1), bias=True),
            self.RBD,)
        self.set_reg_op()
        # self.op = lambda x:\
            # torch.cat([self.op1(x)]*A.in_size[2],3)+torch.cat([self.op2(x)]*A.in_size[2],2)
        self.op = lambda x:x

class E2N(BaseOP):
    def __init__(self, shape_in, shape_out, c_in, c_out, type_) -> None:
        super().__init__(shape_in, shape_out, c_in, c_out, type_)
        self._op = nn.Sequential(
                nn.Conv2d(self.c_out, self.c_out, (1, A.in_size[2]), bias=True),
                self.RBD,)
        self.set_reg_op()
        # self.op = lambda x:self._op(x)
        self.op = lambda x:x

class N2G(BaseOP):
    def __init__(self, shape_in, shape_out, c_in, c_out, type_) -> None:
        super().__init__(shape_in, shape_out, c_in, c_out, type_)
        self._op = nn.Sequential(
                nn.Conv2d(self.c_out, self.c_out, (A.in_size[2], 1), bias=True),
                self.RBD,)
        self.set_reg_op()
        # self.op = lambda x:self._op(x)
        self.op = lambda x:x

