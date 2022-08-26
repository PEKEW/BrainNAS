from Utils import *
from torch.utils.tensorboard import SummaryWriter
from extra import *

def loss_(loss_cal, out, tar):
    a_lable = 0.3
    a_exp = 0.1
    loss = a_lable * loss_cal(out[:,0], tar[:,0])
    for i in range(tar.shape[1]-1):
        loss += a_exp * loss_cal(out[:,i], tar[:,i])
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
    loss_caler = nn.CrossEntropyLoss()
    if set_info['cuda']:
        net = net.cuda()
    net.train()
    # 折
    for fold in data.dataset:
        writer = SummaryWriter()
        # todo 计算模型尺寸
        net_optimizer, path_optimizer = create_optimizer(net)
        train_queue, valid_queue, test_queue = generate_data_queue(fold)
        search_queue = list(zip(train_queue, valid_queue))

        for epoch in range(A.epoch):
            print(f"Seaching ->  {epoch}/{A.epoch}")
            for step, ((train_in, train_tar), (valid_in, valid_tar)) in enumerate(search_queue):
                net_optimizer.zero_grad()
                if path_optimizer: path_optimizer.zero_grad()
                if train_in.shape[0] <= 1 or valid_in.shape[0] <= 1:
                    continue
                if set_info['cuda']:
                    with torch.no_grad():
                        train_in, train_tar = train_in.cuda(), train_tar.cuda()
                        valid_in, valid_tar = valid_in.cuda(), valid_tar.cuda()
                else:
                        train_in, train_tar = V(train_in, requires_grad=False), V(train_tar, requires_grad=False)
                        valid_in, valid_tar = V(valid_in, requires_grad=False), V(valid_tar, requires_grad=False)
                train_out = net(train_in)
                loss = loss_(loss_caler, train_out, train_tar)
                loss.backward()
                # print(loss.data)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=2)
                for name, parameters in net.named_parameters():
                    print(name, 'grad:', parameters.grad)
                net_optimizer.step()
                # path_optimizer.step()
                # for t in range(3):
                # 	dct = {}
                # 	for i in range(10):
                # 		dct[str(t)+'_'+str(i)] = net.path_prob[t,i].data
                # 	writer.add_scalars('path_prob',dct,epoch*100+step)
            pos_ture = 0
            total = 0
            # with TrainMode(net, False) as net:
            #     for _, (test_in, test_tar) in enumerate(test_queue):
            #         if set_info['cuda']:
            #             with torch.no_grad():
            #                 test_in, test_tar = test_in.cuda(), test_tar.cuda()
            #         test_out = net(test_in)
            #         pos_ture += sum((test_out[:,0]>=0.5) & (test_tar[:,0]==1.0)) + \
            #             sum((test_out[:,0] < 0.5) & (test_tar[:,0]==0.0))
            #         # todo 召回率等计算
            #         total += test_out.shape[0]
            #     print(f"acc in [{epoch}] epoch:{pos_ture/total * 100}%")
    
                    


if __name__ == '__main__':
    search()
