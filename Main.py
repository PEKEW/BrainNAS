from Utils import *
from torch.utils.tensorboard import SummaryWriter

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
				path_optimizer.zero_grad()
				if train_in.shape[0] <= 1 or valid_in.shape[0] <= 1:
					continue
				if set_info['cuda']:
					with torch.no_grad():
						train_in, train_tar = train_in.cuda(), train_tar.cuda()
						valid_in, valid_tar = valid_in.cuda(), valid_tar.cuda()
				train_out = net(train_in)
				loss = loss_caler(train_out, train_tar)
				loss.backward()
				# print(loss.data)
				for p in net.parameters():
					print(p.grad)
				net_optimizer.step()
				# path_optimizer.step()
				# for t in range(3):
				# 	dct = {}
				# 	for i in range(10):
				# 		dct[str(t)+'_'+str(i)] = net.path_prob[t,i].data
				# 	writer.add_scalars('path_prob',dct,epoch*100+step)



if __name__ == '__main__':
	search()
