import numpy as np
import pandas as pd
import os
import torch
import torch.backends.cudnn as cudnn
from util import timeit,get_logger
import random
from all_model import AllModel
from process_data import ProcessData,split_train_and_valid,split_train_and_test
from feat import Feat

VERBOSITY_LEVEL = 'INFO'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

import time
s = time.time()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
matrix_a = torch.rand((100,100))
matrix_b = torch.rand((100,100))
torch.mm(torch.Tensor(matrix_a).to(device),torch.Tensor(matrix_b).to(device)).cpu().numpy()
LOGGER.info(f'init torch.mm:{time.time()-s}s')
SEED = 2020

#split_mode = 'stratified','stratified_cv','shuffle_split'
split_mode='stratified_cv'

offline = True
if offline:
    try:
        import prettytable as pt
    except:
        os.system('pip install prettytable')
        import prettytable as pt

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True
    
def cal_wei(acc,degree,has_feature=True):
    degree = np.log(degree)
    if has_feature:
        degree = degree + 1
    degree = np.min([degree,2])
    acc = np.array(acc)
    
    norm = acc - acc.mean()
    norm = (norm-norm.min())
    
    wei = (norm)/((1+degree)**4/8000)+1
    wei = np.exp(wei)/np.sum(np.exp(wei))
    return wei    
    
def fusion(acc_list,run_model,run_type,pred_matrix_list,valid_pred_test_matrixs_list,valid_matrix_list,valid_label,test_label,degree_mean,has_feature):    
    
    weights = cal_wei(acc_list,degree_mean,has_feature=has_feature)
    
    LOGGER.info(f'model_weights:{weights}')
    
    preds = np.zeros(pred_matrix_list[0].shape)
    preds2 = np.zeros(valid_pred_test_matrixs_list[0].shape)
    valid = np.zeros(valid_matrix_list[0].shape)
    for i in range(len(run_model)):
        preds += pred_matrix_list[i]*weights[i]
        preds2 += valid_pred_test_matrixs_list[i]*weights[i]
        valid += valid_matrix_list[i]*weights[i]
    
#    preds = preds*0.8+preds2*0.2
    preds = preds2
    return preds,valid

def fusion2(acc_list,run_model,run_type,pred_matrix_list,valid_pred_test_matrixs_list,valid_matrix_list,valid_label,test_label,degree_mean,has_feature):
    if offline:
        calculateTestAcc(valid_label,test_label,run_model,pred_matrix_list,valid_pred_test_matrixs_list,valid_matrix_list)
    n = len(run_model)
    new_acc_list = []
    new_loss_list = []
    new_run_model = list(set(run_type))
    new_pred_matrix_list = []
    new_valid_pred_test_matrixs_list = []
    new_valid_matrix_list = []
    
    for mode_type in new_run_model:
        tmp_pred = np.zeros(pred_matrix_list[0].shape)
        tmp_valid_pred_test = np.zeros(valid_pred_test_matrixs_list[0].shape)
        tmp_valid = np.zeros(valid_matrix_list[0].shape)
        num = 0
        for i in range(n):
            if mode_type==run_type[i]:
                tmp_pred += pred_matrix_list[i]
                tmp_valid_pred_test += valid_pred_test_matrixs_list[i]
                tmp_valid += valid_matrix_list[i]
                num += 1
        tmp_pred /= num
        tmp_valid_pred_test /= num
        tmp_valid /= num
        
        new_pred_matrix_list.append(tmp_pred)
        new_valid_pred_test_matrixs_list.append(tmp_valid_pred_test)
        new_valid_matrix_list.append(tmp_valid)
        
        acc = (tmp_valid.argmax(axis=1).flatten()==valid_label).mean()
        new_acc_list.append(acc)
        loss = NLLLoss(tmp_valid,valid_label)
        new_loss_list.append(loss)
        
    LOGGER.info(f'model_type_list:{new_run_model}')
    LOGGER.info(f'one type model-valid_acc:{new_acc_list}')
    LOGGER.info(f'one type model-valid_loss:{new_loss_list}')
    if offline:
        calculateTestAcc(valid_label,test_label,new_run_model,new_pred_matrix_list,new_valid_pred_test_matrixs_list,new_valid_matrix_list)
    
    preds,valid = fusion(new_acc_list,new_run_model,new_run_model,new_pred_matrix_list,new_valid_pred_test_matrixs_list,new_valid_matrix_list,valid_label,test_label,degree_mean,has_feature)
    
    return preds,valid

def NLLLoss(matrix,label):
    return np.abs(np.log(matrix)[[range(label.shape[0]),label]]).mean()


def calculateTestAcc(valid_label,test_label,run_model,pred_matrix_list,valid_pred_test_matrixs_list,valid_matrix_list):    
    pred_acc_list = []
    valid_pred_test_acc_list = []
    valid_acc_list = []
    
    pred_loss_list = []
    valid_pred_test_loss_list = []
    valid_loss_list = []
    
    
    for model_num in range(len(pred_matrix_list)):
        pred_acc = (pred_matrix_list[model_num].argmax(axis=1).flatten()==test_label).mean()
        valid_pred_test_acc = (valid_pred_test_matrixs_list[model_num].argmax(axis=1).flatten()==test_label).mean()
        valid_acc = (valid_matrix_list[model_num].argmax(axis=1).flatten()==valid_label).mean()
        
        pred_loss = NLLLoss(pred_matrix_list[model_num],test_label)
        valid_pred_test_loss = NLLLoss(valid_pred_test_matrixs_list[model_num],test_label)
        valid_loss = NLLLoss(valid_matrix_list[model_num],valid_label)
        
        pred_acc_list.append(round(pred_acc,5))
        valid_pred_test_acc_list.append(round(valid_pred_test_acc,5))
        valid_acc_list.append(round(valid_acc,5))
        
        pred_loss_list.append(round(pred_loss,5))
        valid_pred_test_loss_list.append(round(valid_pred_test_loss,5))
        valid_loss_list.append(round(valid_loss,5))
        
    tb = pt.PrettyTable()
    tb.field_names = ["model name", "valid acc","valid preds acc", "test preds acc", "valid loss", "valid preds loss", "test preds loss"]
    for i in range(len(run_model)):
        row = [run_model[i],valid_acc_list[i],valid_pred_test_acc_list[i],pred_acc_list[i],valid_loss_list[i],valid_pred_test_loss_list[i],pred_loss_list[i]]
        tb.add_row(row)
    print(tb)

def get_preds(num,run_model,run_type,allmodel,model_name_list,table,test_label,valid_idx_list):
    valid_idx = valid_idx_list[num]
    pred_matrix_list = []
    valid_pred_test_matrixs_list = []
    valid_matrix_list = []
    acc_list = []
    
    for model_name in run_model:
        pred_matrix_list.append( allmodel.test_pred_matrixs[ model_name ] )
        valid_pred_test_matrixs_list.append( allmodel.valid_pred_test_matrixs[num][ model_name ] )
        valid_matrix_list.append( allmodel.valid_pred_matrixs[num][ model_name ] )
        acc_list.append( allmodel.valid_accs[num][ model_name ] )
        
    
    LOGGER.info(f'model_list:{model_name_list}')
    LOGGER.info(f'model_valid_acc:{allmodel.valid_accs[num]}')
    LOGGER.info(f'model_valid_loss:{allmodel.valid_losses[num]}')
    LOGGER.info(f'learning_rate:{allmodel.learning_rate}')
    
    
    valid_label = table.df.loc[valid_idx,'label'].astype('int').values
    
    degree_mean = table.df['degree'].mean()
    has_feature = True
    if table.ori_columns.shape[0] == 0:
        has_feature = False
    
    preds,valid = fusion2(acc_list,run_model,run_type,pred_matrix_list,valid_pred_test_matrixs_list,valid_matrix_list,valid_label,test_label,degree_mean,has_feature)
    
    valid = valid.argmax(axis=1).flatten()
    valid_acc = (valid_label==valid).sum()/valid.shape[0]
    LOGGER.info(f'valid acc:{valid_acc}')
    
    return preds,valid_acc

class Model:
    @timeit
    def train_predict(self, data, time_budget,n_class,schema):
        s1 = time.time()
        seed = SEED
        fix_seed(seed)
        LOGGER.info(f'time_budget:{time_budget}')
        LOGGER.info(f'n_class:{n_class}')
        LOGGER.info(f'node:{data["fea_table"].shape[0]}')
        LOGGER.info(f'edge:{data["edge_file"].shape[0]}')
        
        #pre-process data
        process_data = ProcessData(data)
        table = process_data.pre_process(time_budget,n_class,schema)
        
        # Feature Dimension Reduction
        feat = Feat()
        
        process_data.drop_unique_columns(table)
        drop_sum_columns = process_data.drop_excessive_columns(table)
        
        feat.fit_transform(table,drop_sum_columns)
        LOGGER.info(f'train:test={(table.df["is_test"]!=1).sum()}:{(table.df["is_test"]==1).sum()}')
        
        #这里好像没用到哦
        table.large_features = False
        if table.ori_columns.shape[0]>500:
            table.large_features = True
        
        model_type_list = ['sage','gat','tagc','gcn']
        
        repeat = 3
        model_name_list = [f'{model_type_list[i]}{i+len(model_type_list)*j}' for j in range(repeat) for i in range(len(model_type_list))]
        model_type_list = model_type_list*repeat
        
        LOGGER.info('use node embedding')
        categories = ['node_index','degree_bins','bin_2-neighbor_mean_degree_bins']        
        
        for model in set(model_type_list):
            LOGGER.info(f"""{model} feature num:{eval(f'table.{model}_columns.shape[0]')}""")
            exec(f'table.{model}_data = process_data.process_gnn_data(table,table.{model}_columns,categories)') 
        
        allmodel = AllModel()
        
        table.lr_epoch = 16
        
        table.lr_list = [0.05,0.03,0.01,0.0075,0.005,0.003,0.001,0.0005]
        
        train_valid_idx_list,valid_idx_list = split_train_and_valid(table,train_rate=0.8,seed=SEED,mode=split_mode)
        train_idx,test_idx = split_train_and_test(table)
        
        test_idx = test_idx.sort_values()
        run_model = []
        run_type = []
        run_time = {}
        for i in range(len(model_type_list)):
            seed = SEED*(i+1)
            fix_seed(seed)
            model_type = model_type_list[i]
            model_name = model_name_list[i]
            if model_type not in run_time:
                init_time, one_epoch_time, early_stopping_rounds = allmodel.get_run_time(table, model_type, model_name, train_idx,test_idx, seed=seed)
                run_lr_time = len(table.lr_list)*(init_time+table.lr_epoch*one_epoch_time)
                run_time500 = init_time*(2)+one_epoch_time*(500+early_stopping_rounds)*2+run_lr_time
                run_time300 = init_time*(2)+one_epoch_time*(300+early_stopping_rounds)*2+run_lr_time
                run_time150 = init_time*(2)+one_epoch_time*(150+early_stopping_rounds)*2+run_lr_time
                run_time[model_type] = (run_time500-run_lr_time,run_time300-run_lr_time,run_time150-run_lr_time,early_stopping_rounds,init_time,one_epoch_time,run_lr_time)
            else:
                run_time500,run_time300,run_time150,early_stopping_rounds,init_time,one_epoch_time,run_lr_time = run_time[model_type]
            s2 = time.time()
            LOGGER.info(f"time_budget:{time_budget}s,used time:{s2-s1:.2f}s,{model_name} model will use {run_time500:.2f}s|{run_time300:.2f}s|{run_time150:.2f}s")
            if s2-s1+run_time500+5<time_budget:
                LOGGER.info('train 500 epoch')
                allmodel.V37_fit_transform(table, model_type, model_name,train_valid_idx_list,valid_idx_list,train_idx,test_idx,mode=split_mode,num_boost_round=500,seed=seed)
                run_model.append(model_name)
                run_type.append(model_type)
            elif s2-s1+run_time300+5<time_budget:
                LOGGER.info('train 300 epoch')
                allmodel.V37_fit_transform(table, model_type, model_name,train_valid_idx_list,valid_idx_list,train_idx,test_idx,mode=split_mode,num_boost_round=300,seed=seed)
                run_model.append(model_name)
                run_type.append(model_type)
            elif s2-s1+run_time150+5<time_budget:
                LOGGER.info('train 150 epoch')
                allmodel.V37_fit_transform(table, model_type, model_name,train_valid_idx_list,valid_idx_list,train_idx,test_idx,mode=split_mode,num_boost_round=150,seed=seed)
                run_model.append(model_name)
                run_type.append(model_type)
            elif len(allmodel.valid_models[0])==0:
                this_epoch = int(((time_budget-(s2-s1+5)-run_lr_time)/2-init_time)/(one_epoch_time)-early_stopping_rounds)
                LOGGER.info(f'short time train {this_epoch} epoch')
                allmodel.V37_fit_transform(table, model_type, model_name,train_valid_idx_list,valid_idx_list,train_idx,test_idx,mode=split_mode,num_boost_round=this_epoch,seed=seed)
                run_model.append(model_name)
                run_type.append(model_type)
            elif time_budget-(s2-s1)<5:
                LOGGER.info('never train; break')
                break
            else:
                LOGGER.info('no train this model; continue')
                continue
        
            
        if offline:
            if table.especial:
                df = table.df[['node_index','is_test']]
                df = df.merge(data['test_label'],how='left',on='node_index')
                test_label = df.loc[(df['is_test']==1)&(table.directed_mask.tolist()),'label'].astype('int').values     
            else:
                test_label = data['test_label']['label'].values
        else:
            test_label = None
            
        
        preds1,valid_acc1 = get_preds(0,run_model,run_type,allmodel,model_name_list,table,test_label,valid_idx_list)
        preds2,valid_acc2 = get_preds(1,run_model,run_type,allmodel,model_name_list,table,test_label,valid_idx_list)
        preds = (preds1+preds2)/2
        
        preds = preds.argmax(axis=1).flatten()
        
            
        if table.especial:
            LOGGER.info(f'preds\n{preds}')
            df = table.df[['label','is_test']]
            df['preds'] = int(df.loc[[not i for i in table.directed_mask.tolist()],'label'].value_counts().index[0])
            df.loc[(df['is_test']==1)&(table.directed_mask.tolist()),'preds'] = preds
            preds = df.loc[df['is_test']==1,'preds'].values
        
        LOGGER.info(f"train label\n{data['train_label']['label'].value_counts()/data['train_label'].shape[0]}")
        df_preds = pd.Series(preds,name='preds')
        LOGGER.info(f"preds label\n{df_preds.value_counts()/df_preds.shape[0]}")
        
        if offline:
            preds1 = preds1.argmax(axis=1).flatten()
            preds2 = preds2.argmax(axis=1).flatten()
            if table.especial:
                LOGGER.info(f'preds1\n{preds1}')
                df = table.df[['label','is_test']]
                df['preds'] = int(df.loc[[not i for i in table.directed_mask.tolist()],'label'].value_counts().index[0])
                df.loc[(df['is_test']==1)&(table.directed_mask.tolist()),'preds'] = preds1
                preds1 = df.loc[df['is_test']==1,'preds'].values
                
                LOGGER.info(f'preds2\n{preds2}')
                df = table.df[['label','is_test']]
                df['preds'] = int(df.loc[[not i for i in table.directed_mask.tolist()],'label'].value_counts().index[0])
                df.loc[(df['is_test']==1)&(table.directed_mask.tolist()),'preds'] = preds2
                preds2 = df.loc[df['is_test']==1,'preds'].values
            
            df_test = table.df[['degree','label','is_test']]
            df_test = df_test.loc[df_test['is_test']==1]
            df_test['preds'] = preds
            df_test['label'] = data['test_label']['label'].values
            df_test['acc'] = df_test['preds']==df_test['label']
            
            pd.set_option('display.max_rows', 1000)
            print(df_test.groupby('degree')['acc'].mean())
            
            return preds,valid_acc1,valid_acc2,preds1,preds2
        else:
            return preds
