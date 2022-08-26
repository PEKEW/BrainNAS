import torch
from torch.optim import Optimizer
from torch.autograd import Variable as V
class TendencyOPT(Optimizer):
    
    def __init__(self, params: V, default: dict) -> None:
        super().__init__(params, default)
        self.lr = default['lr'] if 'lr' in default else 1e-2
        self.momentum = default['momentum'] if 'momentum' in default else 0
        self.weight_decay = default['weight_decay'] if 'weight_decay' in default else 0
        self.gamma = default['gamma'] if 'gamma' in default else 0.01
        self.alpha = default['alpha'] if 'alpha' in default else 0.5
        # self.beta = default['beta'] if 'beta' in default else 0.5
        self.beta = default['beta'] if 'beta' in default else 0
        self.grad_bak = {}
    
    def step(self, closure=None):
        for param_group in self.param_groups:
            params = param_group['params']
            for param in params:
                if param.grad is not None:
                    # print(param.grad.data)
                    z = param.data.clone().detach() - self.lr * param.grad.data.clone()
                    v = self.St(z) - param.data.clone().detach()
                    if param in self.grad_bak:
                        t = param.grad.data.clone() - self.grad_bak[param]
                        self.grad_bak[param] = param.grad.data.clone()
                    else:
                        t = param.grad.data.clone()
                    param.data = param.data + self.alpha * v - self.beta * t
    
    def St(self, data):
        # return torch.sign(data) * torch.max(torch.zeros(data.shape), torch.abs(data) - self.gamma)
        return data