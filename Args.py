class args:
    lr = 0.01
    gpu_id = 0
    weight_decay = 3e-4
    # 网络形状
    in_size = (32,1,200,200) # B C H W
    out_size = 8
    # cell 个数
    cell_num = 3
    # cell中节点的个数
    node_num = 4
    # 数据集类型
    data_type = 'ADHD'
    

    # dropout
    drop_prob = 0.33
    # channles设置
    channles = {
        'e2e': (1, 4),
        'e2n': (8, 16),
        'n2g': (32, 64)
    }
    # 线性层的输入
    liner_in = channles['n2g'][1]
    
    # 形状约束  
    # dict{key: ((in_shape), (out_shape))}
    # in/out_shape = C,W,H
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
    op_leak_relu = 0.33
    
    entropy_weight = 0.33
    
    num_workers = 0
    
    epoch = 50
    
    momentum = 0.9
    
    # 指示用那种算子
    reg_op = {
        'c': 'Copy',
        'h': 'Copy',
        'w': 'Copy',
    }
    # data_path = '/home/scroot/CODE/bNAS/data/Datasets_cc200_10/'
    data_path = '/Users/mac/Desktop/BrainNAS/BrainNAS/data/Datasets_cc200_10/'