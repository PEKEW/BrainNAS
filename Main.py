from Utils import *
from torch.utils.tensorboard import SummaryWriter
from extra import *
import os

def infer(epoch, set_info, net, _queue):
    """计算准确率 召回率等指标

    Args:
        epoch (int): 打印信息用
        set_info (dict): 包含cuda是否可用等信息
        net (M): 模型
        _queue (DataSet): 需要计算指标的推理队列

    Returns:
        Union[int, list]: 计算结果
    """
    correct = 0
    total = 0
    # with TrainMode(net, False) as net:
    net.eval()
    for _, (test_in, test_tar) in enumerate(_queue):
        if set_info['cuda']:
            with torch.no_grad():
                test_in, test_tar = test_in.cuda(), test_tar.cuda()
        test_out, _ = net(test_in)
        _, prediction = torch.min(test_out.data, 1)
        total += test_tar.shape[0]
        correct += prediction.eq(test_tar[:,0].data).sum()
        # todo 召回率等计算
    print(f"acc in [{epoch}] epoch:{correct/total * 100}%")
    return correct/total * 100

def loss_(loss_cal, out, sup_out, tar):
    """计算损失

    Args:
        loss_cal (nn.Loss): 损失计算器
        out (tensor): 网络的推理结果
        sup_out (tensor): 网络的辅助推理结果
        tar (tensor): 标签

    Returns:
        loss: 计算得到的损失
    """
    # tar_onehot = torch.ones(out.shape)
    # for i in range(tar.shape[0]):
    #     # 1 -> 10
    #     # 0 -> 01
    #     if tar[i] == 0:
    #         tar_onehot[i,1] = 0
    #     else:
    #         tar_onehot[i,0] = 0
    loss = loss_cal(out, tar)
    return loss
    

def search():
    # todo 加入对时间的计算
    set_info = search_ready()
    print(set_info['info'])
    if not set_info['is_success']:
        print("please check your enviorment")
        exit(0)
    train(set_info)

def train(set_info):
    data = get_data(A.data_type)
    net = get_net()
    # loss_caler = nn.MSELoss()
    loss_caler = nn.CrossEntropyLoss()
    if set_info['cuda']:
        net = net.cuda()
    net.train()
    # 折
    writer = SummaryWriter()
    for fold in data.dataset:
        best_model = A.model_pth+'_'+str(fold)+'_.pt'
        try:
            os.remove(best_model)
        except FileNotFoundError:
            print("没有在对应路径发现预先保存的模型")
        finally:
            print("预先保存的模型已清除")
        best_acc = -99
        # todo 计算模型尺寸
        # todo 换简单标准网络
        net_optimizer,lr_scheduler, path_optimizer = create_optimizer(net)
        train_queue, valid_queue, test_queue = generate_data_queue(fold)
        queue_list = [train_queue, valid_queue, test_queue]
        search_queue = list(zip(train_queue, valid_queue))

        for epoch in range(A.epoch):
            print(f"Seaching ->  {epoch}/{A.epoch}")
            for step, ((train_in, train_tar), (valid_in, valid_tar)) in enumerate(search_queue):
                net_optimizer.zero_grad()
                if path_optimizer: path_optimizer.zero_grad()
                # todo 直接用tensor
                with torch.no_grad():
                    if set_info['cuda']:
                        train_in, train_tar = train_in.cuda(), train_tar.cuda()
                        valid_in, valid_tar = valid_in.cuda(), valid_tar.cuda()
                    train_in, train_tar = V(train_in), V(train_tar)
                    valid_in, valid_tar = V(valid_in), V(valid_tar)
                train_out, sup_out = net(train_in)
                loss = loss_(loss_caler, train_out, sup_out, train_tar)
                loss.backward()
                print(loss.data)
                # torch.nn.utils.clip_grad_norm_(net.parameters(), A.grad_clip)
                net_optimizer.step()
                lr_scheduler.step()
                # path_optimizer.step()
                # for n,p in net.named_parameters():
                #     if p.grad != None:
                #         print(f"{n}:{p.grad.data}")
                
            acc = []
            for queue_ in queue_list:
                acc.append(infer(epoch, set_info, net, queue_))
            # writer.add_scalars("acc",{
            #     'train':acc[0],
            #     'valid':acc[1],
            #     'test':acc[2]
            # },epoch)
            # if acc[0] > best_acc:
            #     net.eval()
            #     torch.save(net.state_dict(), best_model)
            # else:
            #     net.load_state_dict(torch.load(best_model))

if __name__ == '__main__':
    search()
