# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np
from torch.autograd import Variable


class FeatureEmb(nn.Module):
    def __init__(self,compname, num_team, num_dim, emb_hid=6,flag=0):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(FeatureEmb, self).__init__()
        self.emb_hid = emb_hid
        self.flag = flag
        
        feature_list = [222,27,373,283,26,7]
        self.emb = nn.ModuleList([nn.Embedding(feature_size, emb_hid) for feature_size in feature_list])
        
        self.T_static_emb_tmp = torch.zeros(num_team, num_dim*emb_hid).cuda()
        self.T_static_emb_layer = nn.Embedding(10, emb_hid)
        nn.init.xavier_uniform_(self.T_static_emb_layer.weight.data, gain=math.sqrt(2.0))
        for ele in self.emb:
            nn.init.xavier_uniform_(ele.weight.data, gain=math.sqrt(2.0))
        
    def forward(self, T_static,U_static,team_user_matrix):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps, num_features=in_channels).
        """
        '''static feature embd'''
        TN, TF = T_static.size()
        UN, UF = U_static.size()
        
        U_static_emb = torch.cat([emb(U_static[:,i].long()) for i,emb in enumerate(self.emb)],dim=-1)

        T_static_emb = torch.cat([torch.mean(U_static_emb[np.where(team_user_matrix[i,:]==1),:], dim=1) for i in range(TN)],dim=0)
        
        X_static_emb = torch.cat([T_static, T_static_emb],dim=-1)
        
        if self.flag == 0:
            return X_static_emb
        elif self.flag == 1:
            return T_static_emb
        elif self.flag == 2:
            return T_static


class MTRL(nn.Module):
    '''
    Hierarchical(Recursive) Multi-task RNN with:
    1) both static and dynamic features.
    '''
    def __init__(self,compname,num_timesteps_input,num_timesteps_output,\
                 train_num,num_team,u_static_dim,t_static_dim,dynamic_dim,\
                 dropout=0,tmp_hid=32,static_emb_hid=6,device=torch.device('cpu')):
        super(MTRL, self).__init__()
        self.device = device
        self.tmp_hid = tmp_hid
        self.train_num = train_num
        self.dropout = nn.Dropout(dropout)
        self.t_out = num_timesteps_output
        
        # model block
        # feature embed block
        self.feature_embedding = FeatureEmb(compname,num_team,static_emb_hid,static_emb_hid)
        self.ndynamic_feat = dynamic_dim
        self.nstatic_feat = t_static_dim + u_static_dim * static_emb_hid
        # temporal block
        self.dfea_fc = nn.Linear(self.ndynamic_feat, tmp_hid, bias=True)
        self.GRU = nn.GRUCell(tmp_hid, tmp_hid, bias=True)
        # multi-task block
        # MT level 1: each time step out has an individual block
        self.tmp_fcs = nn.ModuleList([nn.Linear(tmp_hid+self.nstatic_feat, tmp_hid, bias=True) for _ in range(num_timesteps_output)])
        # MT level 2: each task has an individual block
        self.rank_output_fc = nn.Linear(tmp_hid, 1, bias=True)
        self.score_output_fc = nn.Linear(tmp_hid, 1, bias=True)
        self.trend_output_fc = nn.Linear(tmp_hid, 3, bias=True)
        
        # init params
        nn.init.xavier_uniform_(self.GRU.weight_ih,gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.GRU.weight_hh,gain=math.sqrt(2.0))
        for ele in self.modules():
            if isinstance(ele, nn.Linear):
                nn.init.xavier_uniform_(ele.weight,gain=math.sqrt(2.0))

    def forward(self, X_dynamic, T_static,U_static, team_user_matrix, memory, memory_idxs):
        B,N,T,F = X_dynamic.size()
        X_static_emb = self.feature_embedding(T_static,U_static, team_user_matrix)
        X_static_emb = X_static_emb.unsqueeze(0).repeat(B,1,1) # (B,N,F_s)
        h_t = torch.zeros(B*N,self.tmp_hid).to(device=self.device)
        '''GRU'''
        for i in range(T):
            X_feat = self.dfea_fc(X_dynamic[:,:,i,:]) # (B, N, tmp_hid)
            h_t = self.GRU(X_feat.view(B*N,-1), h_t) # (B*N, tmp_hid)
        
        '''concatenate dynamic embedding and static embedding'''
        tmp_out = torch.cat([h_t.view(B,N,-1), X_static_emb],dim=-1)

        '''low level block'''
        h_tmps = torch.cat([tmp_fc(tmp_out).unsqueeze(2) for tmp_fc in self.tmp_fcs],dim=2)# (B, N, T_out,F)
        
        '''high level block'''
        rank_out = torch.cat([torch.sigmoid(self.rank_output_fc(h_tmps[:,:,i,:])) for i in range(self.t_out)],dim=-1) # (B, N, T_out)
        score_out = torch.cat([torch.sigmoid(self.score_output_fc(h_tmps[:,:,i,:])) for i in range(self.t_out)],dim=-1) # (B, N, T_out)
        trend_out = torch.cat([torch.softmax(self.trend_output_fc(h_tmps[:,:,i:i+1,:]), dim=-1) for i in range(self.t_out)],dim=2) # (B, N, T_out,F)
        return rank_out, score_out, trend_out





