from Args import args as A
import scipy.io as io
# 3:Age 4:Sex 5:Handedness 6:FIQ 7:VIQ 8:PIQ 9:EYE Status
path = A.data_path
file_head = 'ALLASD'
file_tail = '_cc200'
folds = 10
key = ['train','valid','test']
print(f"对{path}内数据进行归一化处理")
for i in range(1, folds+1):
    print(f"正在处理第{i}折数据")
    file_name = path+file_head+str(i)+file_tail+'.mat'
    for k in key:
        print(f"正在处理{k}部分的数据")
        data = io.loadmat(file_name)
        label = data['phenotype_'+k][:,2]
        label[label == label.max()] = 1.0
        label[label == label.min] = 0.0
        prest = data['phenotype_'+k][:,3:]
        for j in range(prest.shape[1]):
            prest[:,j] = (prest[:,j] - prest[:,j].min()) / (prest[:,j].max() - prest[:,j].min())
        data['phenotype_'+k][:,2] = label
        data['phenotype_'+k][:,3:] = prest
        print(f"第{i}折数据的第{k}部分处理完毕")
    io.savemat(file_name[:-4]+'_normal.mat',data)
    print(f"第{i}折数据处理完毕")
print(f"{path}内数据全部处理完毕\n ---------------------------")