import numpy as np
from util import timeclass,timeit,get_logger
from table import Table
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold

VERBOSITY_LEVEL = 'INFO'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)

class ProcessData:
    def __init__(self,data):
        self.data = data
    
    @timeclass('ProcessData')
    def pre_process(self,time_budget,n_class,schema):
        df = self.data['fea_table'].copy()
        n = df.shape[0]
        ori_columns = df.columns.drop('node_index')
        df['label'] = df['node_index'].map(self.data['train_label'].set_index('node_index').to_dict()['label'])
        df['is_test'] = 0
        df.loc[df['label'].isnull(),'is_test'] = 1
        
        df_edge = self.data['edge_file'].copy()
        
        edge_matrix = np.identity(n)#添加自环
        edge_matrix[df_edge['src_idx'],df_edge['dst_idx']] = df_edge['edge_weight']
        
        undirected_graph = True if(edge_matrix.T!=edge_matrix).sum()==0 else False
        
        sparse = True if n**2*0.002>df_edge.shape[0] else False
        
        table = Table(df,df_edge,edge_matrix,undirected_graph,sparse,ori_columns,time_budget,n_class,schema,self.data)
        
        return table
    
    @timeclass('ProcessData')
    def drop_unique_columns(self,table):
        #阈值筛选
        df = table.df
        
        drop_columns = df.nunique()
        drop_columns = drop_columns[drop_columns==1].index
                
        table.df = df.drop(drop_columns, axis=1)
        table.ori_columns = table.ori_columns.drop(drop_columns)
        
        for model in table.gnn_list:
            exec(f'table.{model}_columns = table.{model}_columns.drop(drop_columns)')    
            
        LOGGER.info(f'Drop {len(drop_columns)} cols')
    
    @timeclass('ProcessData')
    def drop_excessive_columns(self,table):
        #阈值筛选
        ori_columns = table.ori_columns
        columns_n = ori_columns.shape[0]
        save_columns = 2000
        threshold = 0.99
        drop_sum_columns = pd.Index([])
        if columns_n>=save_columns:
            df = table.df
            df_np = df[ori_columns].values
            if (df_np==0).sum()>df_np.shape[0]*df_np.shape[1]/2:
                sparse_rate = pd.Series((df_np==0).sum(axis=0),index=ori_columns)
            else:
                LOGGER.info('not most 0')
                sparse_rate = (df[ori_columns] == df[ori_columns].mode().iloc[0]).sum(axis=0)
                
            sparse_rate.sort_values(ascending=False,inplace=True)
            drop_columns = sparse_rate[sparse_rate>df.shape[0]*threshold].head(columns_n-save_columns).index
            
            LOGGER.info(f'Drop {len(drop_columns)} cols')
            table.df['drop_sum'] = df[drop_columns].sum(axis=1)
            drop_sum_columns = pd.Index(['drop_sum'])
            
            table.df = df.drop(columns=drop_columns)
            table.ori_columns = table.ori_columns.drop(drop_columns)
            
            for model in table.gnn_list:
                exec(f'table.{model}_columns = table.{model}_columns.drop(drop_columns)')   
        return drop_sum_columns
        
    @timeclass('ProcessData')
    def process_gnn_data(self,table,columns,categories=['node_index']):
        data = self.data
        df = table.df[columns]
        
        #孤立点的id emb做个mask
        df.loc[table.df['degree']==0,'node_index'] = df['node_index'].max()+1
        LOGGER.info(f'Mask isolate point {(table.df["degree"]==0).sum()}')
        
        print(columns)
        
        node_one_hot = torch.tensor(pd.get_dummies(df['node_index']).to_numpy(), dtype=torch.float32)

        categories_value = torch.tensor(df[categories].to_numpy(),dtype=torch.long)
        if categories==[]:
            categories_nums = []
        else:
            categories_nums = (df[categories].max()+1).astype('int').to_list()
        
        df = table.df[columns]
        
        #get label y
        y = torch.zeros(df.shape[0], dtype=torch.long)
        inds = df.loc[df['is_test']==0,'node_index'].to_numpy()
        train_y = df.loc[df['is_test']==0,'label'].to_numpy()
        y[inds] = torch.tensor(train_y, dtype=torch.long)
        
        df = get_model_input(df,categories,label=True)

        df = df.to_numpy()
        df = (df-df.mean(axis=0))/(df.std(axis=0)+1e-5)
        df = np.clip(df,-3,3)
        df = torch.tensor(df, dtype=torch.float)
        
        #get edge_index, edge_weight
        df_edge = table.df_edge.copy()
        edge_index = df_edge[['src_idx', 'dst_idx']].to_numpy()
#        edge_index = sorted(edge_index, key=lambda d: d[0])
        edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
        edge_weight = df_edge['edge_weight'].to_numpy()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
        #get data
        data = Data(x=df, edge_index=edge_index, y=y, edge_weight=edge_weight)
        data.node_one_hot = node_one_hot
        
        data.categories = categories
        data.categories_value = categories_value
        data.categories_nums = categories_nums
        
        data.no_feature = True if table.ori_columns.size==0 else False
        
        return data
    
    @timeit
    def mask_especial_data(self,table):
        df = table.df
        if 'out_degree' in df.columns and 'in_degree' in df.columns:
            if df.loc[df['out_degree'] <= df['in_degree'],'label'].nunique()==1:
                table.especial = True
                #不训练这部分数据
                table.directed_mask[df.loc[df['out_degree'] <= df['in_degree']].index] = 0
                #不使用这些边
                df_edge = table.df_edge
                df_edge = df_edge[df_edge['src_idx'].isin(df.loc[df['out_degree'] > df['in_degree'],'node_index'])].reset_index(drop=True)
                
                df_edge2 = df_edge.copy()
                df_edge2.rename(columns={'src_idx':'dst_idx','dst_idx':'src_idx'},inplace=True)
                df_edge = pd.concat([df_edge,df_edge2],axis=0)
                table.df_edge = df_edge
    
@timeit
def get_model_input(df,categories=[],label=False):
    drop_columns = list(set(['node_index','is_test']+categories))
    if label:
        drop_columns.append('label')
    return df.drop(drop_columns,axis=1)

@timeit
def split_train_and_valid(table,train_rate=0.7,seed=None,mode='stratified'):
#    mode = 'stratified','stratified_cv,'shuffle_split'
    #划分训练集和验证集
    if table.especial:
        df = table.df[table.directed_mask.tolist()]
    else:
        df = table.df
        
    df = df.loc[df['is_test']==0]
    
    train_idx_list = []
    valid_idx_list = []
    
    if mode =='stratified':
        #线上训练集和测试集严格按照类别做的等比例划分，所以验证集的划分方式最好也改，这样特别对齐
        sss=StratifiedShuffleSplit(n_splits=1,test_size=1-train_rate,train_size=train_rate,random_state=seed)
    
        for train_index, test_index in sss.split(df,df['label']):
            train_idx_list.append(df.iloc[train_index])
            valid_idx_list.append(df.iloc[test_index].index.sort_values())
            
    elif mode == 'stratified_cv':
        n_splits = max(3,int(1/(1-train_rate)))
        skf = StratifiedKFold(n_splits=n_splits,shuffle=False,random_state=seed)
        for train_index, test_index in skf.split(df,df['label']):
            train_idx_list.append(df.iloc[train_index].index)
            valid_idx_list.append(df.iloc[test_index].index.sort_values())
        
    elif mode == 'shuffle_split':
        #正常划分
        df = df.sample(frac=1,random_state=seed,axis=0)
        n = int(df.shape[0]*train_rate)
        train_idx_list.append(df[:n].index)
        valid_idx_list.append(df[n:].index.sort_values())
    
    return train_idx_list,valid_idx_list

@timeit
def split_train_and_test(table,seed=None):
    if table.especial:
        df = table.df[table.directed_mask.tolist()]
    else:
        df = table.df
        
    train_idx = df.loc[df['is_test']==0].index
    test_idx = df.loc[df['is_test']==1].index
    return train_idx,test_idx

class ModelData:
    def __init__(self):
        pass
    
    @timeclass('ModelData')
    def split_train_and_valid(self,table,train_rate=0.7,seed=None):
        #划分训练集和验证集
        if table.especial:
            df = table.df[table.directed_mask.tolist()]
        else:
            df = table.df
            
        df = df.loc[df['is_test']==0]
        
        #线上训练集和测试集严格按照类别做的等比例划分，所以验证集的划分方式最好也改，这样特别对齐
        sss=StratifiedShuffleSplit(n_splits=1,test_size=1-train_rate,train_size=train_rate,random_state=seed)

        for train_index, test_index in sss.split(df,df['label']):
            train_idx = df.iloc[train_index].index
            valid_idx = df.iloc[test_index].index
        
        return train_idx,valid_idx
    
    @timeclass('ModelData')
    def split_train_and_test(self,table,seed=None):
        if table.especial:
            df = table.df[table.directed_mask.tolist()]
        else:
            df = table.df
            
        train_idx = df.loc[df['is_test']==0].index
        test_idx = df.loc[df['is_test']==1].index
        return train_idx,test_idx