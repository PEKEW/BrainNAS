from Utils import *
from torch.utils.tensorboard import SummaryWriter
from extra import *

def infer(epoch, set_info, net, test_queue):
    correct = 0
    total = 0
    net.eval()
    for _, (test_in, test_tar) in enumerate(test_queue):
        if set_info['cuda']:
            with torch.no_grad():
                test_in, test_tar = test_in.cuda(), test_tar.cuda()
        test_out, _ = net(test_in)
        _, prediction = torch.max(test_out.data, 1)
        total += test_tar.shape[0]
        correct += prediction.eq(test_tar.data).sum()
        # todo 召回率等计算
    print(f"acc in [{epoch}] epoch:{correct/total * 100}%")
    return correct/total * 100

def loss_(loss_cal, out, sup_out, tar):
    # a_lable = 0.3
    # a_exp = 0.1
    # loss_label = a_lable * loss_cal(out[:,0], tar[:,0])
    # loss_exp = 0
    # for i in range(tar.shape[1]-1):
    #     loss_exp += a_exp * loss_cal(out[:,i], tar[:,i])
    # for so in sup_out:
    #     loss_label += a_lable * loss_cal(so[:,0], tar[:,0])
    #     for i in range(tar.shape[1]-1):
    #         loss_exp += a_exp * loss_cal(so[:,i], tar[:,i])
    # return loss_label + loss_exp/(tar.shape[1]-1)
    tar_onehot = torch.ones(out.shape)
    for i in range(tar.shape[0]):
        if tar[i] == 0:
            tar_onehot[i,1] = 0
        else:
            tar_onehot[i,0] = 0
    loss = loss_cal(out, tar_onehot)
    # for so in sup_out:
    #     loss += loss_cal(so, tar_onehot)
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
    torch.autograd.set_detect_anomaly(True)
    data = get_data(A.data_type)
    net = get_net()
    loss_caler = nn.MSELoss()
    if set_info['cuda']:
        net = net.cuda()
    net.train()
    # 折
    writer = SummaryWriter()
    for fold in data.dataset:
        # todo 计算模型尺寸
        net_optimizer, path_optimizer = create_optimizer(net)
        train_queue, valid_queue, test_queue = generate_data_queue(fold)
        search_queue = list(zip(train_queue, valid_queue))

        for epoch in range(A.epoch):
            print(f"Seaching ->  {epoch}/{A.epoch}")
            for step, ((train_in, train_tar), (valid_in, valid_tar)) in enumerate(search_queue):
                net_optimizer.zero_grad()
                if path_optimizer: path_optimizer.zero_grad()
                # if train_in.shape[0] <= 1 or valid_in.shape[0] <= 1:
                #     continue
                if set_info['cuda']:
                    with torch.no_grad():
                        train_in, train_tar = train_in.cuda(), train_tar.cuda()
                        valid_in, valid_tar = valid_in.cuda(), valid_tar.cuda()
                else:
                        train_in, train_tar = V(train_in, requires_grad=False), V(train_tar, requires_grad=False)
                        valid_in, valid_tar = V(valid_in, requires_grad=False), V(valid_tar, requires_grad=False)
                train_out, sup_out = net(train_in)
                loss = loss_(loss_caler, train_out, sup_out, train_tar)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 20)
                print(loss.data)
                net_optimizer.step()
                # path_optimizer.step()
            train_acc = infer(epoch, set_info, net, train_queue)
            valid_acc = infer(epoch, set_info, net, valid_queue)
            test_acc = infer(epoch, set_info, net, test_queue)
            writer.add_scalars("acc",{
                'train':train_acc,
                'valid':valid_acc,
                'test':test_acc
            },epoch)
            # print(f"acc in {epoch} epoch: {acc}")

if __name__ == '__main__':
    search()
