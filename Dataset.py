import torch
from torch.utils.data import Dataset
import scipy.io as io
import numpy as np
from torch.autograd import Variable as V
from Args import args as A

D = Dataset
T = torch.tensor

class BrainDataBase(D):
    def __init__(self,x,y) -> None:
        x = np.reshape(x,(x.shape[0],1,A.in_size[2],A.in_size[3]))
        self.x = V(T(x,dtype=torch.float32))
        self.y = V(T(y,dtype=torch.float32))
    def __len__(self) -> int:
        return len(self.x)
    def __getitem__(self, index)->list:
        return self.x[index], self.y[index]
    
class BrainDataSet:
    def __init__(self,pth_head:str=None,file_head:str='fold',file_tail:str=None,folds=5) -> None:
        self.pth_head = pth_head or ''
        self.file_head = file_head
        self.folds = folds
        self.tail = file_tail or ''
        self.dataset = self.init_dataset()
    
    def init_dataset(self) -> list:
        key = ['train','valid','test']
        data_set_list = []
        print("loadding data...")
        for i in range(1,self.folds+1):
            data_sets = {}
            file_name = self.pth_head+self.file_head+str(i)+self.tail+'_normal.mat'
            for k in key:
                x = io.loadmat(file_name)['net_'+k]
                y = io.loadmat(file_name)['phenotype_'+k][:,2::]
                data_sets[k] = BrainDataBase(x,y)
            data_set_list.append(data_sets)
            print("all data has been loaded")
            return data_set_list
        
class ABIDE1(BrainDataSet):
    def __init__(self,path:str):
        # super().__init__(pth_head=path,file_head='ALLASD',file_tail='_NETFC_SG_Pear')
        super().__init__(pth_head=path,file_head='ALLASD',file_tail='_cc200',folds=10)
