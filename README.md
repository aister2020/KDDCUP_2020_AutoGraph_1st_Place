# KDD CUP 2020: AutoGraph
### Team: aister

### Introduction
***
+ The competition inviting participants deploy AutoML solutions for graph representation learning, where node classification is chosen as the task to evaluate the quality of learned representations. There are 15 graph datasets which consists of five public datasets to develop AutoML solutions, five feedback datasets to evaluate solutions and other five unseen datasets for the final rankings. Each dataset contains the index value of the node, the processed characteristic value, and the weight of the directed edge. We proposed automatic solutions that can effectively and efficiently learn high-quality representation for each node based on the given features, neighborhood and structural information underlying the graph. Please refer to the competitive official website for more details: https://www.automl.ai/competitions/3

### Preprocess
***
+ Feature
    + The size of node degree can obviously represent the importance of nodes, but the information of node degree with too much value is easy to over fit. So we design the node degree feature of the bucket.
    + Node index value embedding
    + The multi-level neighbor information of the node.

### Model Architecture & Parameters
***
+ We implement four different models:
    + GCN[1]
        + GCN is a basic model in spectral domain. It is easy to extract features from graph data.
            + The embedded node ID, numerical features and category features through embedding layer are passing through dropout layer, linear layer and ELU layer respectively. Concat them as input to the first GCN layer.
            + Pass dropout layer, GCN layer, ELU layer and LN layer twice.
            + Finally, through the dropout layer, linear layer and softmax changes to get the output.
        + Parameters
            + ID_EMBED_SIZE = 8
            + EMBED_SIZE = 8
            + GCN_LAYERS = 2
            + HIDDEN_SIZE = 64
            + DROPOUT_P = 0.1
    + GAT[2]
        + The attention mechanism is added to GAT model. Help the model learn structural information.
        + Parameters
            + ID_EMBED_SIZE = 8
            + EMBED_SIZE = 8
            + GAT_LAYERS = 2
            + HEADS = 2
            + HIDDEN_SIZE = 16
            + DROPOUT_P = 0.1
    + GraphSAGE[3]
        + GraphSage samples the neighbor vertices of each vertex in the graph and aggregates the information contained in the neighbor vertices according to the aggregation function.
        + Parameters
            + ID_EMBED_SIZE = 8
            + EMBED_SIZE = 8
            + GRAPHSAGE_LAYERS = 2
            + HIDDEN_SIZE = 64
            + DROPOUT_P = 0.1
    + TAGConv[4]
        + Accumulate multiple GCN, and get the final result.
            + Similar to GCN model structure, but only through one layer of TAG layer.
        + Parameters
            + ID_EMBED_SIZE = 16
            + EMBED_SIZE = 8
            + TAG_LAYERS = 1
            + K = 3
            + HIDDEN_SIZE = 32
            + DROPOUT_P = 0.1

+ We design different network structures for directed graph and undirected graph, sparse graph and dense graph, graph with node features and graph without node features.

### Training Procedure
***
+ Find Learning Rate
    + lr_list = [0.05, 0.03, 0.01, 0.0075, 0.005, 0.003, 0.001, 0.0005]
    + Select the optimal learning rate of each model in this data set. After 16 rounds of training, choose the learning rate which get lowest loss(average of epoch 14th, 15th and 16th) in the model.
+ Estimated model run time
    + By running the model, estimating the model initialization time and training time each epoch.
    + The model training epochs are determined according to the remaining time and  the running time of the model.
+ Training
    + Because of the small number of model nodes, the difference of several training epochs will lead to the big difference of model effect. Especially for the graph with only node ID information and no original features, it is very easy to over fit. So we adopt cross validation and early stop, which provides us with stable validation effect.
    + training with the following parameters:
        + Learning rate = best_lr
        + Loss: NLL Loss
        + Optimizer: Adam

### Reproducibility
***
+ Requirement
    + `Python==3.6`
    + `torch==1.4.0`
    + `torch-geometric==1.3.2`
    + `numpy==1.18.1`
    + `pandas==1.0.1`
    + `scikit-learn==0.19.1`
+ Training
    + Run `ingestion.py`.

### Reference
***
[1] Kipf T N, Welling M. Semi-supervised classification with graph convolutional networks[J]. arXiv preprint arXiv:1609.02907, 2016.  
[2] Veličković P, Cucurull G, Casanova A, et al. Graph attention networks[J]. arXiv preprint arXiv:1710.10903, 2017.  
[3] Hamilton W, Ying Z, Leskovec J. Inductive representation learning on large graphs[C]//Advances in neural information processing systems. 2017: 1024-1034.  
[4] Du J, Zhang S, Wu G, et al. Topology adaptive graph convolutional networks[J]. arXiv preprint arXiv:1710.10370, 2017.