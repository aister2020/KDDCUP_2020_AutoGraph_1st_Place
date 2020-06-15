import pandas as pd
import numpy as np
from util import timeclass,timeit,get_logger
import torch

VERBOSITY_LEVEL = 'INFO'
LOGGER = get_logger(VERBOSITY_LEVEL, __file__)

@timeit
def sparse_dot(matrix_a,matrix_b):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.mm(torch.Tensor(matrix_a).to(device),torch.Tensor(matrix_b).to(device)).cpu().numpy()


class Feat:
    def __init__(self):
        pass
    
    @timeclass('Feat')
    def fit_transform(self,table,drop_sum_columns):
        degree_columns = self.degree(table)
        degree_bins_columns = self.degree_bins(table)
        
        neighbor_columns = self.get_neighbor(table)
        bin_2_neighbor_mean_degree_bins_columns = self.bin_2_neighbor_mean_degree_bins(table)
        
        gnn_append = [degree_bins_columns,bin_2_neighbor_mean_degree_bins_columns]
        
        #sage feature
        table.sage_columns = table.sage_columns.append(gnn_append)
        
        #gat feature
        table.gat_columns = table.gat_columns.append(gnn_append)
        
        #tagc feature
        table.tagc_columns = table.tagc_columns.append(gnn_append)

        #gcn feature
        table.gcn_columns = table.gcn_columns.append(gnn_append)
        
    @timeclass('Feat')
    def degree(self,table):
        old_columns = table.df.columns
        df = table.df
        df_edge = table.df_edge
        
        if table.undirected_graph:
            LOGGER.info('Undirected graph')
            df['degree'] = df['node_index'].map(df_edge.groupby('src_idx').size().to_dict())
            df['degree'].fillna(0,inplace=True)
        else:
            LOGGER.info('Directed graph')
            df['out_degree'] = df['node_index'].map(df_edge.groupby('src_idx').size().to_dict())
            df['in_degree'] = df['node_index'].map(df_edge.groupby('dst_idx').size().to_dict())
            df['out_degree'].fillna(0,inplace=True)
            df['in_degree'].fillna(0,inplace=True)
            df['degree'] = df['out_degree']+df['in_degree']
            
            df['equal_out_in'] = 0
            df['out_more_than_in'] = 0
            df['in_more_than_out'] = 0
            df.loc[df['out_degree'] == df['in_degree'],'equal_out_in'] = 1
            df.loc[df['out_degree'] > df['in_degree'],'out_more_than_in'] = 1
            df.loc[df['out_degree'] < df['in_degree'],'in_more_than_out'] = 1
            
            LOGGER.info(f"out degree == in degree\n{df.loc[df['equal_out_in']==1,'label'].value_counts()}")
            LOGGER.info(f"out degree > in degree\n{df.loc[df['out_more_than_in']==1,'label'].value_counts()}")
            LOGGER.info(f"out degree < in degree\n{df.loc[df['in_more_than_out']==1,'label'].value_counts()}")
            
            #这里对特殊的数据集做处理，特重要
            if df.loc[df['out_degree'] <= df['in_degree'],'label'].nunique()==1:
                LOGGER.info(f'mask especial unbalanced data')
                table.especial = True
                #不训练这部分数据
                mask_index = df.loc[df['out_degree'] <= df['in_degree']].index
                table.directed_mask[mask_index] = 0
                
                df_tmp = df.loc[table.directed_mask.tolist()]
                LOGGER.info(f'after mask, {df_tmp.shape[0]} points, train:test = {(df_tmp["is_test"]==0).sum()}:{(df_tmp["is_test"]==1).sum()}')
                #不使用这些边
                df_edge = table.df_edge
                df_edge = df_edge[(df_edge['src_idx'].isin(df_tmp['node_index']))].reset_index(drop=True)
                LOGGER.info(f'after mask, {df_edge.shape[0]} edges')
                
                df_edge2 = df_edge.copy()
                df_edge2.rename(columns={'src_idx':'dst_idx','dst_idx':'src_idx'},inplace=True)
                df_edge = pd.concat([df_edge,df_edge2],axis=0)
                table.df_edge = df_edge
            
        table.df = df
        return df.columns.drop(old_columns)
    
    @timeclass('Feat')
    def degree_bins(self,table):
        old_columns = table.df.columns
        df = table.df
        
        bins = int(max(30,df['degree'].nunique()/10))
        
        df_tmp = df['degree'].value_counts().reset_index()
        df_tmp = df_tmp.rename(columns={'index':'degree','degree':'nums'})
        df_tmp = df_tmp.sort_values('degree')
        
        min_nums = df.shape[0]/bins
        k = 0
        cum_nums = 0
        bins_dict = {}
        for i,j in zip(df_tmp['degree'],df_tmp['nums']):
            cum_nums += j
            bins_dict[i] = k
            if cum_nums>=min_nums:
                k += 1
                cum_nums = 0
            
        df['degree_bins'] = df['degree'].map(bins_dict)
        
        table.df = df
        return df.columns.drop(old_columns)
    
    @timeclass('Feat')
    def bin_2_neighbor_mean_degree_bins(self,table):
        old_columns = table.df.columns
        df = table.df
        
        if not table.undirected_graph:
            df['2-neighbor_mean_degree_bins'] = df['2_out-neighbor_mean_degree_bins']+df['2_in-neighbor_mean_degree_bins']
        
        bins = int(min(100,(df['2-neighbor_mean_degree_bins']/0.1).astype(int).nunique()))
        
        df_tmp = df['2-neighbor_mean_degree_bins'].value_counts().reset_index()
        df_tmp = df_tmp.rename(columns={'index':'degree','2-neighbor_mean_degree_bins':'nums'})
        df_tmp = df_tmp.sort_values('degree')
        
        min_nums = df.shape[0]/bins
        k = 0
        cum_nums = 0
        bins_dict = {}
        for i,j in zip(df_tmp['degree'],df_tmp['nums']):
            cum_nums += j
            bins_dict[i] = k
            if cum_nums>=min_nums:
                k += 1
                cum_nums = 0
            
        df['bin_2-neighbor_mean_degree_bins'] = df['2-neighbor_mean_degree_bins'].map(bins_dict)
        
        if not table.undirected_graph:
            df = df.drop(columns='2-neighbor_mean_degree_bins')
        
        table.df = df
        return df.columns.drop(old_columns)
    
    @timeclass('Feat')
    def neighbor(self,edge_matrix,df_value,columns_name,mean=None,namepsace=None):
        df_neighbor = pd.DataFrame(sparse_dot(edge_matrix,df_value),columns=[f'{namepsace}-neighbor_sum_{col}' for col in columns_name])
        if mean is not None:
            df_neighbor = pd.DataFrame(df_neighbor.to_numpy()*(1/(mean.to_numpy()+1)).reshape(-1,1),columns=df_neighbor.columns)
            df_neighbor.rename(columns={f'{namepsace}-neighbor_sum_{col}':f'{namepsace}-neighbor_mean_{col}' for col in columns_name},inplace=True)
            
        return df_neighbor
    
    @timeclass('Feat')
    def get_neighbor(self,table):
        old_columns = table.df.columns
        columns_name = [i for i in table.df.columns if i not in ['node_index','is_test','label']]
        edge_matrix = table.edge_matrix
        df = table.df
        df_value = df[columns_name]
        if table.undirected_graph:
            df_neighbor = self.neighbor(edge_matrix,df_value.values,columns_name,mean=df['degree'],namepsace=1)
            df_neighbor2 = self.neighbor(edge_matrix,df_neighbor.values,columns_name,mean=df['degree'],namepsace=2)
            
            if table.unbalanced:
                df_neighbor3 = self.neighbor(edge_matrix,df_neighbor2.values,columns_name,mean=df['degree'],namepsace=3)
                df = pd.concat([df,df_neighbor,df_neighbor2,df_neighbor3],axis=1)
            else:
                df = pd.concat([df,df_neighbor2],axis=1)
        else:
            out_df_neighbor = self.neighbor(edge_matrix,df_value.values,columns_name,mean=df['out_degree'],namepsace='1_out')
            out_df_neighbor2 = self.neighbor(edge_matrix,out_df_neighbor.values,columns_name,mean=df['out_degree'],namepsace='2_out')
            
            in_df_neighbor = self.neighbor(edge_matrix.T,df_value.values,columns_name,mean=df['in_degree'],namepsace='1_in')
            in_df_neighbor2 = self.neighbor(edge_matrix.T,in_df_neighbor.values,columns_name,mean=df['in_degree'],namepsace='2_in')
            
            df = pd.concat([df,out_df_neighbor2,in_df_neighbor2],axis=1)
        
        table.df = df
        return df.columns.drop(old_columns)
    