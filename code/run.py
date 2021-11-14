# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pickle
import glob
import numpy as np
import torch
import time
import torch.nn as nn
import logging
import random
from models import *
from utils import *

which_gpu = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = which_gpu
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
parser = argparse.ArgumentParser(description='Run the MTRL model')
parser.add_argument('--enable_cuda', action='store_true', default=True, help='Disables CUDA training.')
parser.add_argument('--compname', type=str, default='KDD', help='competition name.')
parser.add_argument('--model', type=str, default='MTRL', help='name of baseline model.')
parser.add_argument('--seed', type=int, default=33, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=12, help='Number of batch to train and test.')
parser.add_argument('--t_in', type=int, default=12, help='Input time step.')
parser.add_argument('--t_out', type=int, default=3, help='Output time step.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--static_emb_hid', type=int, default=6, help='Number of hidden units for embedding block.')
parser.add_argument('--tmp_hidden', type=int, default=32, help='Number of hidden units for temporal block.')
parser.add_argument('--train_num', type=int, default=-1, help='lots num for train.')
parser.add_argument('--train_ratio', type=float, default=0.3, help='lots num for train.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=200, help='Patience')
parser.add_argument('--lambda1', type=float, default=0.25, help='multitask parameter')
parser.add_argument('--train_percent', type=float, default=0.6, help='training set ratio.')
parser.add_argument('--test_part', type=str, default='final', help='training set ratio.')
args = parser.parse_args()

HOME = '/data3/guoqingyu/TKDD-KDD_Cup'
logging.basicConfig(level = logging.INFO,filename=HOME+'/log/{}-{}-[{}]-{}dim-{}lr-{}t_in-{}t_out-{}wd-{}lambda.log'.\
                                    format(args.compname, args.model, args.seed,args.tmp_hidden,args.lr,args.t_in,args.t_out,args.weight_decay,args.lambda1),\
                    format = '%(asctime)s - %(process)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
                                     

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.enable_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
print(args)
logger.info(args)

def train_epoch(loader_train,T_static,U_static,team_user_matrix,memory,memory_idx_train):
    for i,(X_batch,Y_batch) in enumerate(loader_train):
        if((i+1)*args.batch_size<len(memory_idx_train)):
            memory_idxs = memory_idx_train[i*args.batch_size:(i+1)*args.batch_size]
        else:
            memory_idxs = memory_idx_train[i*args.batch_size:]
        net.train()
        X_batch = X_batch.to(device=args.device)
        Y_batch = Y_batch.to(device=args.device)
        optimizer.zero_grad()
        y_pred_rank, y_pred_score, y_pred_trend = net(X_batch,T_static,U_static,team_user_matrix,memory,memory_idxs) # (B, N , T_out)
        rank_loss = loss_criterion(y_pred_rank, Y_batch[:,:,:,0])
        score_loss = loss_criterion(y_pred_score, Y_batch[:,:,:,1])
        trend_loss = loss_criterion_trend(y_pred_trend.view(-1,3), Y_batch[:,:,:,-1].view(-1).long())
        loss = 0.5*rank_loss + args.lambda1*score_loss + args.lambda1*trend_loss
        loss.backward()
        optimizer.step()
        if (i*args.batch_size % 100 == 0):
            print("train loss:{:.4f},{:.4f},{:.4f}"
                  .format(loss.detach().cpu().numpy(),
                          rank_loss.detach().cpu().numpy(),
                          score_loss.detach().cpu().numpy(),
                         )
                 )
    
    return loss.detach().cpu().numpy(), rank_loss.detach().cpu().numpy(),\
           score_loss.detach().cpu().numpy(), trend_loss.detach().cpu().numpy()


def test_epoch(loader_val,T_static,U_static,team_user_matrix,memory,memory_idx_val):
    val_loss = []
    val_rank_loss = []
    val_score_loss = []
    val_trend_loss = []
    val_rank_mae = []
    val_score_mae = []
    val_trend_mae = []
    ndcg_s = []
    for i,(X_batch,Y_batch) in enumerate(loader_val):
        if((i+1)*args.batch_size<len(memory_idx_val)):
            memory_idxs = memory_idx_val[i*args.batch_size:(i+1)*args.batch_size]
        else:
            memory_idxs = memory_idx_val[i*args.batch_size:]
        net.eval()
        X_batch = X_batch.to(device=args.device)
        Y_batch = Y_batch.to(device=args.device)
        y_pred_rank, y_pred_score, y_pred_trend = net(X_batch,T_static,U_static,team_user_matrix,memory,memory_idxs) # (B,N,T_out)
        loss_val_rank = loss_criterion(y_pred_rank, Y_batch[:,:,:,0])
        loss_val_score = loss_criterion(y_pred_score, Y_batch[:,:,:,1])

        loss_val_trend = loss_criterion_trend(y_pred_trend.view(-1,3), Y_batch[:,:,:,-1].view(-1).long())
        loss_val = 0.5*loss_val_rank + args.lambda1*loss_val_score + args.lambda1*loss_val_trend

        val_loss.append(np.asscalar(loss_val.detach().cpu().numpy()))
        val_rank_loss.append(np.asscalar(loss_val_rank.detach().cpu().numpy()))
        val_score_loss.append(np.asscalar(loss_val_score.detach().cpu().numpy()))
        val_trend_loss.append(np.asscalar(loss_val_trend.detach().cpu().numpy()))

        rank_mae = np.absolute(y_pred_rank.detach().cpu().numpy() * T_num - Y_batch[:,:,:,0].detach().cpu().numpy() * T_num) # (B,N,T_out)
        score_mae = np.absolute(y_pred_score.detach().cpu().numpy() * T_num - Y_batch[:,:,:,1].detach().cpu().numpy() * T_num) # (B,N,T_out)
        val_rank_mae.append(rank_mae)
        val_score_mae.append(score_mae)

        for i in range(args.t_out):
            temp_list = []
            for j in range(Y_batch.size()[0]):

                real_k = np.argsort(np.argsort(Y_batch[j,:,i,0].detach().cpu().numpy())).tolist()
                pred_k = np.argsort(np.argsort(y_pred_rank[j,:,i].detach().cpu().numpy()))
                pred_k_order = np.argsort(pred_k).tolist()
                
                DCG = 0

                tmp_total = len(pred_k_order)
                for r in range(len(pred_k_order)):
                    temp_score = np.exp2(len(pred_k_order) - real_k[pred_k_order[r]] - 1)-1
                    
                    temp_score = float(temp_score / np.log2(r + 2))
                    DCG += temp_score
                
                ICG = 0
                for r in range(len(real_k)):
                    temp_score = np.exp2(len(real_k) - r - 1)-1
                    temp_score = float(temp_score / np.log2(r + 2))
                    ICG += temp_score
                NDCG = DCG / ICG
                temp_list.append(NDCG)
            temp_mean = np.mean(temp_list)
            ndcg_s.append(temp_mean)
        
    print('ndcg_s: ',ndcg_s)
    return np.asarray(val_loss),np.asarray(val_rank_loss),np.asarray(val_score_loss),np.asarray(val_trend_loss),\
           np.concatenate(val_rank_mae,axis=0),np.concatenate(val_score_mae,axis=0), ndcg_s

        
def print_log(mae,mse,loss,stage, ndcg):

    mae_o = [np.mean(mae[:,:,i]) for i in range(mae.shape[-1])]
    mse_o = [np.mean(mse[:,:,i]) for i in range(mse.shape[-1])]
    rmse_o = [np.sqrt(ele) for ele in mse_o]
    stage_str = "{} - mean metrics: mae,mse,rmse,loss, ndcg"\
                .format(stage,np.mean(mae_o),np.mean(mse_o),np.mean(rmse_o),np.mean(loss))
    mean_str = "mean metric values: {},{},{},{},{}".format(np.mean(mae_o),np.mean(mse_o),np.mean(rmse_o),np.mean(loss),np.mean(ndcg))
    mae_str = "MAE: {}".format(','.join(str(ele) for ele in mae_o))
    mse_str = "MSE: {}".format(','.join(str(ele) for ele in mse_o))
    rmse_str = "RMSE: {}".format(','.join(str(ele) for ele in rmse_o))
    ndcg_str = "NDCG: {}".format(','.join(str(ele) for ele in ndcg))
    print(stage_str)
    print(mean_str)
    print(mae_str)
    print(mse_str)
    print(rmse_str)
    print(ndcg_str)
    logger.info(stage_str)
    logger.info(mean_str)
    logger.info(mae_str)
    logger.info(mse_str)
    logger.info(rmse_str)
    logger.info(ndcg_str)

def cal_rel(pos, total, bins):
    propotion = float(pos)/total
    partial = int(propotion * bins)
    return bins-partial
        
if __name__ == '__main__':
    team_user_matrix,loader_train,loader_val,loader_test,T_static,U_static, \
    D_dim, memory, memory_idx_train, memory_idx_val, memory_idx_test = \
            load_data(args.t_in,args.t_out,args.batch_size,args.compname,args.tmp_hidden, train_percent=args.train_percent, test_part=args.test_part)
    U_num, U_dim = U_static.shape
    T_num, T_dim = T_static.shape
    T_static = T_static.to(device=args.device)
    U_static = U_static.to(device=args.device)
    memory = memory.to(device=args.device)

    
    
    net = MTRL(compname=args.compname,
            num_timesteps_input=args.t_in,
            num_timesteps_output=args.t_out,
            train_num = args.train_num,
            num_team = T_num,
            u_static_dim = U_dim,
            t_static_dim = T_dim,
            dynamic_dim = D_dim,
            dropout=args.dropout, 
            tmp_hid=args.tmp_hidden,
            static_emb_hid=args.static_emb_hid,
            device=args.device).to(device=args.device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_criterion = nn.L1Loss()
    loss_criterion_trend = nn.NLLLoss()
    
    min_mae = 1e15
    best_epoch = 0
    for epoch in range(args.epochs):
        st_time = time.time()
        '''training'''
        print('training......')
        loss_train, rank_loss_train, score_loss_train, trend_loss_train = \
            train_epoch(loader_train,T_static,U_static,team_user_matrix,memory,memory_idx_train)
        '''validating'''
        with torch.no_grad():
            print('validating......')
            val_loss,val_rank_loss, val_score_loss, val_trend_loss, val_rank_mae, val_score_mae, val_ndcg = test_epoch(loader_val,T_static,U_static,team_user_matrix,memory,memory_idx_val)
            val_rank_mse = val_rank_mae**2
            val_score_mse = val_score_mae**2
        '''testing'''
        with torch.no_grad():
            print('testing......')
            test_loss, test_rank_loss, test_score_loss, test_trend_loss, test_rank_mae, test_score_mae, test_ndcg = test_epoch(loader_test,T_static,U_static,team_user_matrix,memory,memory_idx_test)
            test_rank_mse = test_rank_mae**2
            test_score_mse = test_score_mae**2
            
        val_meanmae = np.mean(val_rank_mae)
        if(val_meanmae < min_mae):
            min_mae = val_meanmae
            best_epoch = epoch + 1
            best_rank_mae = test_rank_mae.copy()
            best_rank_mse = test_rank_mse.copy()
            best_rank_loss = test_rank_loss.copy()
            best_ndcg = test_ndcg


        try:
            print("Epoch: {}".format(epoch+1))
            logger.info("Epoch: {}".format(epoch+1))
            print("Train loss: {}".format(loss_train))
            print("Train rank loss: {}".format(rank_loss_train))
            print("Train score loss: {}".format(score_loss_train))
            logger.info("Train loss: {}".format(loss_train))
            logger.info("Train rank loss: {}".format(rank_loss_train))
            logger.info("Train score loss: {}".format(score_loss_train))
            print_log(val_rank_mae,val_rank_mse,val_rank_loss,'Validation',val_ndcg)

            print_log(test_rank_mae,test_rank_mse,test_rank_loss,'Test',test_ndcg)
            print_log(best_rank_mae,best_rank_mse,best_rank_loss,'Best Epoch-{}'.format(best_epoch), best_ndcg)
            print('time: {:.4f}s'.format(time.time() - st_time))
            logger.info('time: {:.4f}s\n'.format(time.time() - st_time))
        except:
            pass
        
        # model stop condition
        if(epoch+1 - best_epoch >= args.patience):
            sys.exit(0)
            
                
               
