import torch

class Table:
    def __init__(self,df,df_edge,edge_matrix,undirected_graph,sparse,ori_columns,time_budget,n_class,schema,data):
        self.df = df
        self.df_edge = df_edge
        self.edge_matrix = edge_matrix
        self.undirected_graph = undirected_graph
        self.sparse = sparse
        self.categories = []
        self.time_budget = time_budget
        self.n_class = n_class
        self.schema = schema
        
        self.ori_columns = ori_columns
        self.lgb_columns = df.columns.copy()
        
        self.gnn_list = ['sage','gat','tagc','gcn']
        
        for model in self.gnn_list:
            exec(f'self.{model}_columns=df.columns.copy()')
            
        self.especial = False
        self.directed_mask = torch.ones(df.shape[0],dtype=torch.bool)
        
        self.unbalanced = False
        label_value = df['label'].value_counts()/df.loc[df['is_test']==0].shape[0]
        if label_value.shape[0]==3 and label_value.iloc[0]>0.45 and label_value.iloc[1]>0.45:
            self.unbalanced = True
            