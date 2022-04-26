import numpy as np
import pandas as pd
import torch

def list():
    return ["test1","time1"]

def dictionary():
    return {"test1":Test1(),
            "time1":Test_shift_over_time_1()}

class Test1():
    def __init__(self):

        self.path = "SavedModels/test1"
        self.encoder_shape = [8,16,16,8]
        self.decoder_shape = [8,16,16,8]
        
        self.shift = [0,0,0,0,0,0,0,2]

        data=np.random.normal(0,1,(100000,8))
        data_raw=pd.DataFrame(data)
        self.data_A=data_raw[ ::2] - [x/2 for x in self.shift]
        self.data_B=data_raw[1::2] + [x/2 for x in self.shift]

        self.dataloader_train_all,self.dataloader_test_all = Create_Dataloader(pd.concat([self.data_A,self.data_B]))
        self.dataloader_train_A,self.dataloader_test_A = Create_Dataloader(self.data_A)
        self.dataloader_train_B,self.dataloader_test_B = Create_Dataloader(self.data_B)

class Test_shift_over_time_1():
    def __init__(self):
        
        self.path = "SavedModels/time1"
        self.encoder_shape = [9,16,16,8]
        self.decoder_shape = [9,16,16,8]
        
        self.shift = [0,0,0,0,0,0,0,1]
        N = 100000
        data=np.random.normal(0,1,(N,8))
        time = np.random.uniform(0,1,(N,1))
        shift_time = time*self.shift
        data[1::2] += shift_time[1::2]
        data = np.append(data,time,axis=1)
        data_raw=pd.DataFrame(data)

        self.data_A=data_raw[ ::2] 
        self.data_B=data_raw[1::2]

        self.dataloader_train_all,self.dataloader_test_all = Create_Dataloader(pd.concat([self.data_A,self.data_B]))
        self.dataloader_train_A,self.dataloader_test_A = Create_Dataloader(self.data_A)
        self.dataloader_train_B,self.dataloader_test_B = Create_Dataloader(self.data_B)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df,isVal=False, val_split=0.1):
        self.data = torch.from_numpy(df.values).float()
        assert val_split>=0. and val_split<=1., "Val split should be between 0 and 1"
        if isVal:
            if val_split == 0.:
                self.data = torch.empty(0)
            else:
                val_stride = int(1/val_split)
                self.data = self.data[::val_stride]
        else:
            if val_split > 0.:
                mask=torch.ones(len(self.data),dtype=torch.bool)
                mask[torch.arange(start=0, end=len(self.data), step=int(1/val_split))]=False
                self.data = self.data[mask]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item

def Create_Dataloader(data):
    ds_train = CustomDataset(data,False)
    ds_test = CustomDataset(data,True)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=10000, shuffle=True)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=len(ds_test), shuffle=True)
    return dl_train,dl_test