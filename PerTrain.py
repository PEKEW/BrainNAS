from Utils import *
import os

def per_train():
    _info = search_ready()
    print(_info)
    _cuda = _info['cuda']
    data = get_data(A.data_type)
    loss_caler = nn.CrossEntropyLoss()
    net = get_net()
    if _cuda: net = net.cuda()
    net.train()
    for fold_data in data.dataset:
        # todo 保存模型
        _optimizer, scheduler, _ = create_optimizer(net)
        queue_list = generate_data_queue(fold_data)
        train_queue = list(zip(queue_list[0]))
        
        for epoch in range(A.epoch):
            print(f"Training -> {epoch}/{A.epoch}")
            for step, (train_in, train_tar) in enumerate(train_queue):
                _optimizer.zero_grad()
                with torch.no_grad():
                    if _cuda:
                        train_in, train_tar = train_in.cuda(), train_tar.cuda()
                    train_in, train_tar = V(train_in) , V(train_tar)
                train_out = net(train_in)

if __name__ == '__main__':
    per_train()