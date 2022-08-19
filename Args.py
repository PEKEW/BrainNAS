class args:
    lr = 1e-3
    gpu_id = 0
    weight_decay = 3e-4
    # 网络形状
    in_size = (16,1,90,90) # B C H W
    out_size = (8)
    # cell 个数
    cell_num = 3
    # cell中节点的个数
    node_num = 4
    # 数据集类型
    data_type = 'ADHD'
    

    # dropout
    drop_prob = 0.5
    # channles设置
    channles = {
        'e2e': (1, 8),
        'e2n': (32, 64),
        'n2g': (256, 512)
    }
    # 线性层的输入
    liner_in = channles['n2g'][1]
    
    # channles约束
    channles_constraint = {
        'e2e': channles['e2e'][1],
        'e2n': channles['e2n'][1],
        'n2g': channles['n2g'][1]
    }
    # cell的输入输出形状设置
    cell_shape = {
        'e2e': ((90,90), (90,90)),
        'e2n': ((90,90), (90,1)),
        'n2g': ((90,1), (1,1))
    }
    op_leak_relu = 0.33
    
    entropy_weight = 0.33
    
    num_workers = 0
    
    epoch = 50
    
    momentum = 0.9
    
    # 指示用那种算子
    reg_op = {
        'c': 'Conv',
        'h': 'Copy',
        'w': 'Copy',
    }