# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

class GetDataset(Dataset):
    def __init__(self, X, Y):

        self.X = X   # numpy.ndarray (num_data, num_nodes(N), T_in, num_features(F))
        self.Y = Y   # numpy.ndarray (num_data, num_nodes(N), T_out, num_features(F))
        
    def __getitem__(self, index):
        
        # torch.Tensor
        tensor_X = self.X[index]
        tensor_Y = self.Y[index]
        
        return tensor_X, tensor_Y

    def __len__(self):
        return len(self.X)






def make_dataset(rawdata, T_in, T_out):
    X = [] 
    Y = []
    T_all = rawdata.shape[1]
    pdata = rawdata.copy()
  
    for i in range(T_all-(T_in+T_out)):
        X.append(pdata[:, i:i+T_in, :])
        Y.append(rawdata[:, i+T_in:i+(T_in+T_out), :])

    X = torch.from_numpy(np.asarray(X)).float()
    Y = torch.from_numpy(np.asarray(Y)).float().squeeze(-1)

    return GetDataset(X, Y)




def load_data(T_in, T_out, Batch_Size, compname, hid,shuffle_flag=0,train_percent=0.6,test_part='final'):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :return X: (num_data, num_nodes(N), T_in, num_features(F))
    :return Y: (num_data, num_nodes(N), T_out, num_features(F))
    """
    HOME = '/data3/guoqingyu/TKDD-KDD_Cup'
    X_dynamic = np.load(HOME+'/data/{}/X_dynamic_7.npy'.format(compname))
    X_dynamic = X_dynamic[:,:,:] 
    U_static = np.load(HOME+'/data/{}/U_static.npy'.format(compname))
    U_static = torch.from_numpy(np.asarray(U_static)).float()
    T_static = np.load(HOME+'/data/{}/T_static.npy'.format(compname))
    T_static = T_static/float(np.max(T_static))
    T_static = torch.from_numpy(np.asarray(T_static)).float()
    team_user_matrix = np.load(HOME+'/data/{}/team_user_matrix.npy'.format(compname))
    T_all = X_dynamic.shape[1]
    TN = T_static.shape[0]
    UN = U_static.shape[0]
    D_dim = X_dynamic.shape[-1]

    dataset_train = make_dataset(X_dynamic[:,:int(T_all*train_percent)],T_in,T_out)
    dataset_val = make_dataset(X_dynamic[:,int(T_all*train_percent)-(T_in+T_out):int(T_all*0.8)],T_in,T_out)
    if test_part == 'final':
        dataset_test = make_dataset(X_dynamic[:,int(T_all*0.8)-(T_in+T_out):],T_in,T_out)
    elif test_part == 'subseq':
        dataset_test = make_dataset(X_dynamic[:,int(T_all*train_percent)-(T_in+T_out):int(T_all*(0.2+train_percent))],T_in,T_out)
    memory = torch.zeros(T_all, TN, hid, requires_grad=False)
    memory_idx_train = range(T_in-1,int(T_all*train_percent)-T_out-1)
    memory_idx_val = range(int(T_all*train_percent)-T_out-1,int(T_all*0.8)-T_out-1)
    memory_idx_test = range(int(T_all*0.8)-T_out-1,T_all-T_out-1)
    
    loader_train = DataLoader(dataset=dataset_train, batch_size=Batch_Size, shuffle=True, pin_memory=True,num_workers=1)
    loader_val = DataLoader(dataset=dataset_val, batch_size=Batch_Size, shuffle=False, pin_memory=True,num_workers=1)
    loader_test = DataLoader(dataset=dataset_test, batch_size=Batch_Size, shuffle=False, pin_memory=True,num_workers=1)
    
    print("load_data finished.")
    return team_user_matrix,loader_train,loader_val,loader_test,T_static,U_static, D_dim, memory, memory_idx_train, memory_idx_val, memory_idx_test
       
    





