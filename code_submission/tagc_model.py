import torch
import torch.nn.functional as F
from torch.nn import Linear, Embedding
from torch_geometric.nn import TAGConv
from torch_geometric.data import Data

from util import timeclass,get_logger
import pandas as pd
import numpy as np
import random
import time
import copy
from process_data import ModelData

VERBOSITY_LEVEL = 'INFO'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)

class TAGC(torch.nn.Module):

    def __init__(self,categories_nums, features_num=16, num_class=2, sparse=False, degree_mean=2):
        super(TAGC, self).__init__()
        hidden = 32
        embed_size = 8
        dropout = 0.1
        self.dropout_p = dropout
        
        id_embed_size = 16
        
        self.id_embedding = Embedding(categories_nums[0], id_embed_size)
        
        self.lin0_id_emb = Linear(id_embed_size, id_embed_size)
        
        self.embeddings = torch.nn.ModuleList()
        for max_nums in categories_nums[1:]:
            self.embeddings.append(Embedding(max_nums, embed_size))
        
        n = max(0,len(categories_nums)-1)
        if n>0:
            self.lin0_emb = Linear(embed_size*n, embed_size*n)
        
        if sparse:
            if features_num == 0:
                K= max(7,int(np.exp(-(degree_mean-1)/1.5)*100))
            else:
                K=6
        else:
            K=3
        
        LOGGER.info(f'K values:{K}')
        if features_num>0:
            self.lin0 = Linear(features_num, hidden)
            self.ln0 = torch.nn.LayerNorm(id_embed_size+embed_size*n+hidden)
            self.conv1 = TAGConv(id_embed_size+embed_size*n+hidden, hidden,K=K)
        else:
            self.ln0 = torch.nn.LayerNorm(id_embed_size+embed_size*n)
            self.conv1 = TAGConv(id_embed_size+embed_size*n, hidden,K=K)
            
        self.ln1 = torch.nn.LayerNorm(hidden)
        self.lin1 = Linear(hidden, num_class)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        
        if x.shape[1]>0:
            x = F.dropout(x, p=self.dropout_p, training=self.training)
            x = self.lin0(x)
            x = F.elu(x)
        
        id_emb = self.id_embedding(data.categories_value.T[0,:])
        id_emb = F.dropout(id_emb, p=self.dropout_p, training=self.training)
        id_emb = self.lin0_id_emb(id_emb)
        id_emb = F.elu(id_emb)
        
        
        emb_res = []
        for f,emb in zip(data.categories_value.T[1:],self.embeddings):
            emb_res.append(emb(f))
            
        if len(emb_res)>0:
            emb_res = torch.cat(emb_res,axis=1)
            emb_res = F.dropout(emb_res, p=self.dropout_p, training=self.training)
            emb_res = self.lin0_emb(emb_res)
            emb_res = [F.elu(emb_res)]
        
        x = torch.cat( [id_emb,x]+emb_res, axis=1 )
        
        x = self.ln0(x)
        
        x = F.dropout(x, p=self.dropout_p, training=self.training)     
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        
        x = self.ln1(x)
        
        x = self.lin1(x)
        
        return F.log_softmax(x, dim=-1)

class TAGCModel:
    def __init__(self,num_boost_round=1001,best_iteration=100):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = 20
        self.best_iteration = best_iteration
        self.learning_rate = 0.005
        self.echo_epochs = 1
        
        self.before_epochs = 20
        self.after_epochs = 20
    
    @timeclass('TAGCModel')
    def init_model(self,model_data,table):
        data = model_data.tagc_data
        sparse = table.sparse
        categories_nums = data.categories_nums
        
        degree_mean = table.df['degree'].mean()
        model = TAGC(categories_nums,features_num=data.x.size()[1], num_class=table.n_class,sparse=sparse,degree_mean=degree_mean)
        
        model = model.to(self.device)
        data = data.to(self.device)

        LOGGER.info(f'learning rate:{self.learning_rate}')
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=3, verbose=True, threshold=1e-4, threshold_mode='rel', cooldown=5, min_lr=self.learning_rate/100, eps=1e-08)
        
        return model,data,optimizer,scheduler

    @timeclass('TAGCModel')
    def train_and_valid(self, model_data, table, seed=None):
        model,data,optimizer,scheduler = self.init_model(model_data,table)

        pre_time = time.time()
        best_acc = 0
        best_epoch = 0
        best_loss = 1e9
        best_valid_loss = 1e9
        best_model = copy.deepcopy(model)
        best_pred_matrix = None
        keep_epoch = 0
        
        
        acc_list = [0]
        lr_list = [optimizer.param_groups[0]['lr']]
        
        model.train()
        for epoch in range(1,self.num_boost_round+1):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss( output[data.valid_train_mask], data.y[data.valid_train_mask] )
            valid_loss = F.nll_loss( output[data.valid_test_mask], data.y[data.valid_test_mask] )
            acc = (output[data.valid_test_mask].max(1)[1]==data.y[data.valid_test_mask]).sum().float() / len(data.y[data.valid_test_mask])
            
            acc_list.append(acc.item())
            
            if acc > best_acc:
#            if valid_loss<best_valid_loss:
                best_acc = acc
                best_epoch = epoch
                best_loss = loss
                best_valid_loss = valid_loss
                best_model = copy.deepcopy(model)
                best_pred_matrix = output[data.valid_test_mask]
                keep_epoch = 0
            else:
                keep_epoch += 1
                if keep_epoch > self.early_stopping_rounds:
                    now_time = time.time()
                    print(f'epoch:{epoch} [train loss]: {loss.data} [valid loss]: {valid_loss.data} [valid acc]: {acc} [use time]: {now_time-pre_time} s')
                    break
                
            if epoch%self.echo_epochs==0:
                now_time = time.time()
                print(f'epoch:{epoch} [train loss]: {loss.data} [valid loss]: {valid_loss.data} [valid acc]: {acc} [use time]: {now_time-pre_time} s')
                pre_time = now_time
            loss.backward()
            optimizer.step()
#            scheduler.step(acc)
            lr_list.append(optimizer.param_groups[0]['lr'])
        self.model = best_model
        print(f'lr_list:{lr_list}')
        after_epochs = self.after_epochs
        k = 0.03
        threshold = best_acc.item()-k
        for i in range(best_epoch+1,min(best_epoch+self.before_epochs+1,epoch)):
            if acc_list[i]<threshold:
                after_epochs = i-best_epoch-1
                LOGGER.info(f'after {after_epochs} epochs acc reduce over {k}')
                break
            
        return best_epoch, float(best_acc.cpu().numpy()), best_pred_matrix.cpu().detach().exp().numpy(),best_loss,best_valid_loss,after_epochs,lr_list

        
    @timeclass('TAGCModel')
    def train(self, model_data,table,lr_list,seed=None):
        model,data,optimizer,scheduler = self.init_model(model_data,table)
        model.train()
        pre_time = time.time()
        
        first_iteration = max(0,self.best_iteration-self.before_epochs)+1
        for epoch in range(1,first_iteration):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])  
            if epoch%self.echo_epochs==0:
                now_time = time.time()
                print(f'epoch:{epoch} [train loss]: {loss.data}, use time: {now_time-pre_time}s')
                pre_time = now_time
            loss.backward()
            optimizer.param_groups[0]['lr'] = lr_list[epoch-1]
            optimizer.step()
        
        best_loss = 1e9
        best_model = copy.deepcopy(model)
        best_epoch = first_iteration
        for epoch in range(first_iteration,self.best_iteration+self.after_epochs+1):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])  
            if loss<best_loss:
                best_loss = loss
                best_model = copy.deepcopy(model)
                best_epoch = epoch
            if epoch%self.echo_epochs==0:
                now_time = time.time()
                print(f'epoch:{epoch} [train loss]: {loss.data}, use time: {now_time-pre_time}s')
                pre_time = now_time
            loss.backward()
            if epoch<=len(lr_list):
                optimizer.param_groups[0]['lr'] = lr_list[epoch-1]
            optimizer.step()
        
        LOGGER.info(f'best epoch:{best_epoch}, best loss:{best_loss}')
        self.model = best_model
        
        best_model.eval()
        with torch.no_grad():
            preds_matrix = best_model(data)[data.test_mask].cpu().detach().exp().numpy()
#        return output[data.test_mask].cpu().detach().exp().numpy()
        return preds_matrix

    @timeclass('TAGCModel')
    def predict(self, model_data):
        data = model_data.tagc_data
        self.model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            preds_matrix = self.model(data)[data.test_mask].cpu().detach().exp().numpy()
            preds = preds_matrix.argmax(axis=1).flatten()
        return preds,preds_matrix

    @timeclass('TAGCModel')
    def get_run_time(self, model_data, table):
        t1 = time.time()
        model,data,optimizer,scheduler = self.init_model(model_data,table)
        t2 = time.time()
        for epoch in range(1,6):
            model.train()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss( output[data.train_mask], data.y[data.train_mask] )
            loss.backward()
            optimizer.step()
        t3 = time.time()
        init_time = t2-t1
        one_epoch_time = (t3-t2)/5
        LOGGER.info(f'init_time:{init_time},one_epoch_time:{one_epoch_time}')
        return init_time, one_epoch_time

    @timeclass('TAGCModel')
    def get_train_and_valid(self,table,train_valid_idx,valid_idx,seed=None):
        #划分训练集和验证集
        valid_model_data = ModelData()

        #获取gcn数据
        data = table.tagc_data.clone()
        num_nodes = data.y.shape[0]
        
        valid_train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        valid_train_mask[train_valid_idx] = 1
        data.valid_train_mask = valid_train_mask
        
        valid_test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        valid_test_mask[valid_idx] = 1
        data.valid_test_mask = valid_test_mask
        
        valid_model_data.tagc_data = data
        
        return valid_model_data
    
    @timeclass('TAGCModel')
    def get_train(self,table,train_idx,test_idx,seed=None):
        #划分训练集和测试集
        model_data = ModelData()
        
        #获取gcn数据
        data = table.tagc_data.clone()
        num_nodes = data.y.shape[0]
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = 1
        data.train_mask = train_mask
        
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[test_idx] = 1
        data.test_mask = test_mask
        
        model_data.tagc_data = data
        
        return model_data
    
    @timeclass('TAGCModel')
    def get_lr(self,lr_one,model_data,table,seed=None):
        self.learning_rate = lr_one
        model,data,optimizer,scheduler = self.init_model(model_data,table)
        loss_list = np.zeros(table.lr_epoch)
        valid_loss_list = np.zeros(table.lr_epoch)
        acc_list = np.zeros(table.lr_epoch)
        
        LOGGER.info(f'learning rate:{lr_one}')
        model.train()
        for epoch in range(table.lr_epoch):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss( output[data.valid_train_mask], data.y[data.valid_train_mask] )
            valid_loss = F.nll_loss( output[data.valid_test_mask], data.y[data.valid_test_mask] )
            acc = (output[data.valid_test_mask].max(1)[1]==data.y[data.valid_test_mask]).sum().float() / len(data.y[data.valid_test_mask])
            
            loss_list[epoch] = loss.item()
            valid_loss_list[epoch] = valid_loss.item()
            acc_list[epoch] = acc.item()
            
            print(f'[{epoch+1}/{table.lr_epoch}] train loss:{loss.data}, valid loss:{valid_loss.data}, valid acc:{acc.data}')
            loss.backward()
            optimizer.step()
        k = 3
#        return loss.item(),valid_loss.item(),acc.item()
        return loss_list[-k:].mean(),valid_loss_list[-k:].mean(), acc_list[-k:].mean()
