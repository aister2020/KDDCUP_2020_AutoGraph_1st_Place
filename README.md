# KDD CUP 2020: AutoGraph
### Team: aister
***
+ Members: Jianqiang Huang, Xingyuan Tang, Mingjian Chen, Jin Xu, Bohang Zheng, Yi Qi, Ke Hu, Jun Lei
+ Team Introduction: Most of our members come from the Search Ads Algorithm Team of the Meituan Dianping Advertising Platform Department. We participated in three of the five competitions held by KDD CUP 2020 and achieved promising results. We won first place in Debiasing(1/1895), first place in AutoGraph(1/149), and third place in Multimodalities Recall(3/1433).
+ Based on the business scenario of Meituan and Dianping App, the Search Ads Algorithm Team of Meituan Dianping has rich expertise in innovation and algorithm optimization in the field of cutting-edge technology, including but not limited to, conducting algorithm research and application in the fileds of Debiasing, Graph Learning and Multimodalities.
+ If you are interested in our team or would like to communicate with our team(b.t.w., we are hiring), you can email to huangjianqiang@meituan.com.

### Introduction
***
+ The competition inviting participants deploy AutoML solutions for graph representation learning, where node classification is chosen as the task to evaluate the quality of learned representations. There are 15 graph datasets which consists of five public datasets to develop AutoML solutions, five feedback datasets to evaluate solutions and other five unseen datasets for the final rankings. Each dataset contains the index value of the node, the processed characteristic value, and the weight of the directed edge. We proposed automatic solutions that can effectively and efficiently learn high-quality representation for each node based on the given features, neighborhood and structural information underlying the graph. Please refer to the competition official website for more details: https://www.automl.ai/competitions/3

### Preprocess
***
+ Feature
    + The size of node degree can obviously represent the importance of node, but the information of node degree with too much value is easy to overfit. So we bucket the node degree.
    + Node index embedding
    + The multi-hop neighbor information of the node.

### Model Architecture
***
+ Automatic proxy evaluation is a better method to select proper models for a new dataset. However, the extremely limited time budget does not allow online model selection. For a trade-off of accuracy and speed, we offline evaluate many models and empirically find that GCN, GAT, GraphSAGE, and TAGConv can get robust and good results on the 5 public dataset and 5 feedback datasets. Thus we use them for ensemble in this code. One can get better results using proxy evaluation.
+ We design different network structures for directed graph and undirected graph, sparse graph and dense graph, graph with node features and graph without node features.

### Training Procedure
***
+ Search learning rate
    + lr_list = [0.05, 0.03, 0.01, 0.0075, 0.005, 0.003, 0.001, 0.0005]
    + Select the optimal learning rate of each model in this data set. After 16 rounds of training, choose the learning rate which get lowest loss(average of epoch 14th, 15th and 16th) in the model.
+ Estimate running time
    + By running the model, estimating the model initialization time and training time for each epoch.
    + The model training epochs are determined according to remaining time and running time of the model.
+ Training and validation
    + The difference of training epochs will lead to the big difference of model effect. It is very easy to overfit for the graph with only node ID information and no original features. So we adopt cross validation and early stopping, which makes the model more robust.
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
