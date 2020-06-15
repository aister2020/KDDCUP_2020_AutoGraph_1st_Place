from gcn_model import GCNModel
from sage_model import SAGEModel
from gat_model import GATModel
from tagc_model import TAGCModel
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from util import timeclass, get_logger

VERBOSITY_LEVEL = 'INFO'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)

class AllModel:
    def __init__(self):
        self.valid_models = [{},{}]
        self.valid_accs = [{},{}]
        self.valid_losses = [{},{}]
        self.valid_pred_matrixs = [{},{}]
        self.valid_pred_test_matrixs = [{},{}]
        
        self.test_models = {}
        self.test_pred_matrixs = {}
        
        self.valid_model_datas = [{},{}]
        self.test_model_datas = {}
        
        self.learning_rate = {}
    
    @timeclass('AllModel')
    def V37_fit_transform(self,table, model_type, model_name, train_valid_idx_list,valid_idx_list,train_idx,test_idx,mode='stratified',num_boost_round=1001, seed=2020):
        train_valid_idx = train_valid_idx_list[0]
        valid_idx = valid_idx_list[0]
        
        model_valid = eval(f'{model_type.upper()}Model(num_boost_round=num_boost_round)')
        
        if model_type in self.valid_model_datas[0]:
            valid_model_data = self.valid_model_datas[0][model_type]
        else:
            valid_model_data = model_valid.get_train_and_valid(table,train_valid_idx,valid_idx,seed=seed)
            self.valid_model_datas[0][model_type] = valid_model_data
        
        #计算最优初始lr
        if model_type in self.learning_rate:
            lr = self.learning_rate[model_type]
        else:
            lr_loss = []
            lr_valid_loss = []
            lr_acc = []
            for lr_one in table.lr_list:
                model_lr = eval(f'{model_type.upper()}Model()')
                loss, valid_loss, acc = model_lr.get_lr(lr_one,valid_model_data,table,seed=seed)
                lr_loss.append(loss)
                lr_valid_loss.append(valid_loss)
                lr_acc.append(acc)
            lr = table.lr_list[lr_valid_loss.index(min(lr_valid_loss))]
#            lr = table.lr_list[lr_acc.index(max(lr_acc))]
            LOGGER.info(f'lr list:{table.lr_list}')
            LOGGER.info(f'lr loss:{lr_loss}')
            LOGGER.info(f'lr valid loss:{lr_valid_loss}')
            LOGGER.info(f'lr acc:{lr_acc}')
            LOGGER.info(f'best learning rate:{lr}')
            
            self.learning_rate[model_type] = lr
        
        model_valid.learning_rate = lr
        best_epoch, valid_acc, valid_pred_matrix,best_loss,best_valid_loss,after_epochs,lr_list = model_valid.train_and_valid(valid_model_data,table,seed=seed)
        
        self.valid_models[0][model_name] = model_valid
        self.valid_accs[0][model_name] = valid_acc
        self.valid_losses[0][model_name] = best_valid_loss.item()
        self.valid_pred_matrixs[0][model_name] = valid_pred_matrix
        LOGGER.info(f'best epoch:{best_epoch}, best acc:{valid_acc}, best loss:{best_loss}, best valid loss:{best_valid_loss}')
        ##############我是分割线，上面是valid，下面是train####################
        
        if mode=='stratified':
            model = eval(f'{model_type.upper()}Model(best_iteration=best_epoch)')
            
            if model_type in self.test_model_datas:
                model_data = self.test_model_datas[model_type]
            else:
                model_data = model.get_train(table,train_idx,test_idx,seed=seed)
                self.test_model_datas[model_type] = model_data
                
            model.learning_rate = lr
            model.after_epochs = after_epochs
            test_pred_matrix = model.train(model_data,table,lr_list,seed=seed)
            self.test_models[model_name] = model
            self.test_pred_matrixs[model_name] = test_pred_matrix
            
            #valid model 的结果
            test_valid_pred, test_valid_pred_matrix = model_valid.predict(model_data)
            self.valid_pred_test_matrixs[0][model_name] = test_valid_pred_matrix
        elif mode=='stratified_cv':
            train_valid_idx2 = train_valid_idx_list[1]
            valid_idx2 = valid_idx_list[1]
            
            model_valid2 = eval(f'{model_type.upper()}Model(num_boost_round=num_boost_round)')
            
            if model_type in self.valid_model_datas[1]:
                valid_model_data2 = self.valid_model_datas[1][model_type]
            else:
                valid_model_data2 = model_valid2.get_train_and_valid(table,train_valid_idx2,valid_idx2,seed=seed)
                self.valid_model_datas[1][model_type] = valid_model_data2
            
            model_valid2.learning_rate = lr
            best_epoch2, valid_acc2, valid_pred_matrix2,best_loss2,best_valid_loss2,after_epochs2,lr_list2 = model_valid2.train_and_valid(valid_model_data2,table,seed=seed)
            
            if model_type in self.test_model_datas:
                model_data = self.test_model_datas[model_type]
            else:
                model_data = model_valid2.get_train(table,train_idx,test_idx,seed=seed)
                self.test_model_datas[model_type] = model_data
            
            
            self.valid_models[1][model_name] = model_valid2
            self.valid_accs[1][model_name] = valid_acc2
            self.valid_losses[1][model_name] = best_valid_loss2.item()
            self.valid_pred_matrixs[1][model_name] = valid_pred_matrix2
            LOGGER.info(f'best epoch:{best_epoch2}, best acc:{valid_acc2}, best loss:{best_loss2}, best valid loss:{best_valid_loss2}')
            
            self.test_models = model_valid2
            
            #valid model 的结果
            test_valid_pred, test_valid_pred_matrix = model_valid.predict(model_data)
            test_valid_pred2, test_valid_pred_matrix2 = model_valid2.predict(model_data)
            
            self.valid_pred_test_matrixs[0][model_name] = test_valid_pred_matrix
            self.valid_pred_test_matrixs[1][model_name] = test_valid_pred_matrix2
            
            self.test_pred_matrixs[model_name] = (test_valid_pred_matrix+test_valid_pred_matrix2)/2
            
    @timeclass('AllModel')
    def get_run_time(self, table, model_type, model_name, train_idx,test_idx, seed=2020):
        model = eval(f'{model_type.upper()}Model()')
        
        if model_type in self.test_model_datas:
            model_data = self.test_model_datas[model_type]
        else:
            model_data = model.get_train(table,train_idx,test_idx,seed=seed)
            self.test_model_datas[model_type] = model_data
            
        init_time, one_epoch_time = model.get_run_time(model_data,table)
        return init_time, one_epoch_time, model.early_stopping_rounds