class args:
    
    # ! 网络训练相关参数
    # 学习率
    lr = 3e-3
    # 梯度衰减
    weight_decay = 1e-4
    # 梯度动量
    momentum = 0.9
    # GPU ID
    gpu_id = 0
    # 迭代次数
    epoch = 100
    # 梯度裁剪
    grad_clip = 5
    # Leak Relu 参数
    op_leak_relu = 0.33
    # dropout
    drop_prob = 0.3
    # 损失中标签占比
    loss_alpha = 0.3
    # 损失中表现型占比
    loss_exp = 0.1
    
    
    # ! 数据集相关参数
    # 数据集加载时多线程数量
    num_workers = 0
    # 数据集路径
    # data_path = '/home/scroot/CODE/bNAS/data/Datasets_cc200_10/'
    data_path = '/Users/mac/Desktop/BrainNAS/BrainNAS/data/Datasets_cc200_10/'
    file_tail = '_cc200'
    # 数据集类型
    data_type = 'ADHD'
    
    # ! 网络形状相关参数
    # B C H W
    in_size = (96,1,200,200) 
    # 输出个数
    out_size = 2
    # cell 个数
    cell_num = 3
    # cell中节点的个数
    node_num = 4
    # channles设置
    # * 因为有节点拼合 所以上一层的输出和下一层的输入呈倍数关系: 倍数为 node_num
    channles = {
        'e2e': (1, 4),
        'e2n': (8, 16),
        'n2g': (32, 64)
    }
    # 线性层的输入
    liner_in = channles['n2g'][1]
    # 形状约束  
    # * dict{key: ((in_shape), (out_shape))}
    # * in/out_shape = C,W,H
    shape_constraint = {
        'e2e': ((channles['e2e'][1], in_size[2], in_size[3]),(channles['e2e'], in_size[2], in_size[3])),
        'e2n': ((channles['e2n'][1], in_size[2], in_size[3]), (channles['e2n'][1], in_size[2], 1)),
        'n2g': ((channles['n2g'][1], in_size[2], 1), (channles['n2g'][1], 1, 1))
    }
    # cell的输入输出形状设置
    cell_shape = {
        'e2e': ((in_size[2],in_size[3]), (in_size[2],in_size[3])),
        'e2n': ((in_size[2],in_size[3]), ((in_size[2],1))),
        'n2g': (((in_size[2],1)), (1,1))
    }
    # 是否需要辅助头
    need_help = False
    
    # ! 未定参数
    entropy_weight = 0.33
    

