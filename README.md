# SMGF: Spectrum-guided Multi-view Attributed Graph Fusion

This repository contains the implementation of **SMGF**  and **SMGFQ** algorithm.

## Prerequisites

Install dependencies via Conda:
```
conda create --name SGLA --file requirements.txt -c pytorch
conda activate SGLA
```

Extract dataset files: 
```
unzip data.zip
```

## Usage

9 available datasets as follows: 

4 muliplex datasets: DBLP, IMDB, Yelp, RM.

2 graph datasets with mulipile features: Amazon-photos, Amazon-computers.

(tips: 2 large-scale datasets: Mag-eng, Mag-phy will be available later.)

2 Efficient integration function of Multi-View Attributed Graphs are as follows:

**SMGFQ** finds a surrogate objective via quadratic regression for efficient optimization. 

Please choose the one you want to use.

Parameters used:

| Parameter     | Default | Description                                                  |
| ------------- | ------- | ------------------------------------------------------------ |
| --dataset     | dblp    | choosed used dataset                                         |
| --scale       | -       | configurations for large-scale data (mageng/magphy)          |
| --verbose     | -       | produce verbose command-line output                          |
| --embedding   | -       | configure for generating embedding, default clustering       |
| --embed_dim   | 64      | embedding output demension                                   |
| --embed_rank  | 32      | NETMF/SKETCHNE parameter for embedding                       |
| --eig_tol     | 0.01    | precision of eigsh solver                                    |
| --knn_k       | 10      | $K$, k neighbors except imdb=500, yelp=200, query=20         |
| --opt_t_max   | 100     | $T_{max}$, maximum number of iterations for COBYLA optimizer |
| --opt_epsilon | 0.001   | $\epsilon$, convergence threshold for COBYLA optimizer       |
| --obj_gamma   | 0.5     | $\gamma$, coefficient of weight regularization               |
| --ridge_alpha | 0.05    | $a_r$, regularization parameter for ridge regression         |

To **reproduce** the results in our paper, please refer to the following command lines for testing corresponding datasets with your choosed method.
#### **SMGF** Clustering and embedding:
##### DBLP
```
python SMGF.py --dataset dblp
```
##### DBLP
```
python SMGF.py --dataset dblp --embedding
```
##### Yelp
```
python SMGF.py --dataset yelp --knn_k 200
```
##### Yelp
```
python SMGF.py --dataset yelp --knn_k 200 --embedding
```

#### **SGLA+** Clustering and embedding:
##### DBLP
```
python SMGFQ.py --dataset dblp
```
##### DBLP
```
python SMGFQ.py --dataset dblp --embedding
```
##### Yelp
```
python SMGFQ.py --dataset yelp --knn_k 200
```
##### Yelp
```
python SMGFQ.py --dataset yelp --knn_k 200 --embedding
```

Sample output of **SMGF**  for clustering and embedding on Yelp:

`
Acc: 0.930 F1: 0.934 NMI: 0.739 ARI: 0.785 Time: 0.739s RAM: 214MB
`

`
Labeled data 20%: f1_macro: 0.943, f1_micro: 0.938, roc_auc_macro: 0.990, roc_auc_micro: 0.991
Time: 2.379s RAM: 421MB
`

Sample output of **SMGFQ** for clustering and embedding on Yelp:

`
Acc: 0.930 F1: 0.932 NMI: 0.733 ARI: 0.787 Time: 1.178s RAM: 215MB
`

`
Labeled data 20%: f1_macro: 0.942, f1_micro: 0.937, roc_auc_macro: 0.990, roc_auc_micro: 0.991
Time: 1.823s RAM: 419MB
`


