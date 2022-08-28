
from torch.autograd import Variable as V
import torch.nn as nn
import torch

class FuckingGradList(nn.ModuleList):
    def __init__(self, modules=None):
        super().__init__(modules)
    def __getitem__(self, idx: int):
        idx = -1 if idx >= len(self) else idx
        return super().__getitem__(idx)

class FindFuckingGrad(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.op = nn.Sequential(nn.LeakyReLU(0.33),
        nn.BatchNorm2d(1, affine=True),
        nn.Dropout(0.2))
    def forward(self, x):
        return self.op(x)

class FindFuckingGrad2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ops = nn.ModuleList()
        for _ in range(3):
            self.ops.append(
                FindFuckingGrad()
            )
    def forward(self, x):
        return sum(op(x) for op in self.ops)

class FindFuckingGrad3(nn.Module):
    def __init__(self):
        super().__init__()
        self.ops = FuckingGradList()
        for _ in range(10):
            self.ops.append(FindFuckingGrad2())
    
    def forward(self, x):
        s = [x]
        idx = 0
        for _ in range(4):
            h_s = 0
            idx += len(s)
            for i, s_ in enumerate(s):
                h_ = self.ops[idx](s_)
                h_s += h_
            s.append(h_s)
        out = torch.cat(s[1::],dim=1)
        return out

class FindFuckingGrad4(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ops = nn.ModuleList()
        for _ in range(3):
            self.ops.append(FindFuckingGrad3())
    def forward(self, x):
        for op in self.ops:
            x = op(x)
        return x

if __name__ == '__main__':
    i = V(torch.randn(20, 1, 35, 45),requires_grad=False) 
    m = FindFuckingGrad2()
    o = m(i)
    y = V(torch.randn(o.shape), requires_grad=False)
    func = nn.CrossEntropyLoss()
    loss = func(y,o)
    print(f"loss:{loss}")
    loss.backward()
    for name, p in m.named_parameters():
        print(f"name:{name} grad:{p.grad}")
