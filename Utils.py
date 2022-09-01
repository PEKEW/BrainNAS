import random
import torch.cuda as cuda
import torch.backends.cudnn as cudnn
from Args import args as A
from Nets import HyperNet, BrainNetCNN
import torch.nn as nn
from Dataset import *
import torch
from Optimizer import TendencyOPT
from torch.utils.data import DataLoader
DL = DataLoader

T = torch.Tensor
HN = HyperNet
BD = BrainDataSet

def search_ready() -> dict:
    """搜索的预备设置
    return: {
        bool is suuccess
        str  info
        int  seed
    }
    """
    info = "settings:\n"
    seed = random.randint(1,10000)
    info += f"seed: {seed}\n"
    cuda_available = cuda.is_available()
    is_success = True
    
    if cuda_available:
        info += "cuda is availabe\n"
        try:
            cuda.set_device(A.gpu_id)
        except:
            info += "GPU set failed\n"
            is_success = False
        else:
            info += f"GPU:{A.gpu_id}\n"
        cuda.manual_seed(seed)
        cudnn.benchmark = True
        cudnn.enabled = True
        info += "cudnn set success\n"
    else:
        info += "cuda is not available\n"        
    return {
        'is_success': is_success,
        'info': info,
        'seed': seed,
        'cuda': cuda.is_available()
    }
    
def get_net() -> HN:
    """初始化搜索网络
    """

    net = HN()
    # net = BrainNetCNN()
    return net
    
def get_data(type_:str) ->BD:
    """初始化数据集

    Args:
        type_ (数据集类型): ADHD,ABIDE1,ABIDE2
    """
    if type_ == 'ADHD':
        return ABIDE1(A.data_path)
    elif type_ == 'ABIDE1':
        # todo
        pass
    elif type_ == 'ABIDE2':
        # todo
        pass
    else:
        print("dataset type error!")
        exit(0)

def cal_entropy(x) -> T:
    """计算节点的图像熵
    """
    # todo 
    return x
    
def loss(loss_caler:nn.CrossEntropyLoss, y_:T, y:T, entropy:T, path_prob:T) -> T:
    """计算网络损失

    Args:
        loss_caler (nn.CrossEntropyLoss): 计算基础的损失
        y_ (T): 网络输出   
        y (T): 标签
        size(x) == size(y)
        entropy (T): 每个cell的熵 <1x1>
        path_prob (T): 每个cell上每条路径的概率 <cell_numxpath_num>

    Returns:
        T: _description_
    """
    base_loss = loss_caler(y_, y)
    entropy_loss = entropy * A.entropy_weight
    # 先对每个cell的概率求平均 然后在每个cell的尺度上 算L1范数
    sparsity_loss = torch.sum(torch.abs(torch.mean(path_prob, dim=0)))
    return base_loss + entropy_loss + sparsity_loss

def create_optimizer(net) -> list:

    net_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), 
                                lr=A.lr,
                                momentum=A.momentum,
                                weight_decay=A.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(net_optimizer, T_max=150, eta_min=0)
    if isinstance(net ,BrainNetCNN):
        path_optimizer = None
    else:
        path_optimizer = TendencyOPT(net.get_path_prob(), {})
    return net_optimizer, lr_scheduler, path_optimizer

def generate_data_queue(dataset) -> list:
    
    train_data, valid_data, test_data = dataset['train'], dataset['valid'], dataset['test']
        
    batch_size = A.in_size[0]
    num_workers = A.num_workers

    train_queue = DL(
        train_data, batch_size=batch_size,
        pin_memory=True, num_workers=num_workers,shuffle=True)

    valid_queue = DL(
        valid_data, batch_size=batch_size,
        pin_memory=True, num_workers=num_workers,shuffle=True)

    test_queue = DL(
        test_data, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True,shuffle=True)

    return train_queue, valid_queue, test_queue


# todo
def triu(x:T) -> T:
    """把输入取上三角 展开然后返回 同时保证梯度传播

    Args:
        x (T): shape = B C W H

    Returns:
        T: shape = B *
    """
    ...