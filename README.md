# SGLA and SGLA+: Efficient Integration of Multi-View Attributed Graphs for Clustering and Embedding

This repository contains the implementation of **SGLA**  and **SGLA+** algorithm.

## Prerequisites

Install dependencies by `conda create --name <env> --file requirements.txt -c pytorch`.

Unzip the content of "data.zip" into "data" folder by `unzip data.zip` to use datasets.

## Usage

6 available datasets as follows: 

4 muliplex datasets: DBLP, IMDB, Yelp, RM.

2 graph datasets with mulipile features: Amazon-photos, Amazon-computers.

(tips: 2 large-scale datasets: Mag-eng, Mag-phy will be available later.)

2 Efficient integration function of Multi-View Attributed Graphs are as follows:

**SGLA** directly optimizes the objective with a derivative-free iterative method in SGLA.py

**SGLA+** finds a surrogate objective via quadratic regression for efficient optimization in SGLAplus.py

Please choose the one you want to use.

Parameters used:

| Parameter     | Default | Description                                                  |
| ------------- | ------- | ------------------------------------------------------------ |
| --dataset     | dblp    | choosed used dataset                                         |
| --scale       | -       | configurations for large-scale data (mageng/magphy)          |
| --verbose     | -       | produce verbose command-line output                          |
| --embedding   | -       | configure for generating embedding, default clustering       |
| --embed_dim   | 64      | embedding output demension                                   |
| --eig_tol     | 0.01    | precision of eigsh solver                                    |
| --knn_k       | 10      | $K$, k neighbors except imdb=500, yelp=200, query=20         |
| --opt_t_max   | 100     | $T_{max}$, maximum number of iterations for COBYLA optimizer |
| --opt_epsilon | 0.001   | $\epsilon$, convergence threshold for COBYLA optimizer       |
| --obj_gamma   | 0.5     | $\gamma$, coefficient of weight regularization               |
| --ridge_alpha | 0.05    | $a_r$, regularization parameter for ridge regression         |

To **reproduce** the results in our paper, please refer to the following command lines for testing corresponding datasets with your choosed method.
#### **SGLA** Clustering and embedding:
##### DBLP
```
python SGLA.py --dataset dblp
```
##### DBLP
```
python SGLA.py --dataset dblp --embedding
```
##### Yelp
```
python SGLA.py --dataset yelp --knn_k 200
```
##### Yelp
```
python SGLA.py --dataset yelp --knn_k 200 --embedding
```

#### **SGLA+** Clustering and embedding:
##### DBLP
```
python SGLAplus.py --dataset dblp
```
##### DBLP
```
python SGLAplus.py --dataset dblp --embedding
```
##### Yelp
```
python SGLAplus.py --dataset yelp --knn_k 200
```
##### Yelp
```
python SGLAplus.py --dataset yelp --knn_k 200 --embedding
```

Sample output of **SGLA**  for clustering and embedding on Yelp:

`
Acc: 0.927 F1: 0.930 NMI: 0.728 ARI: 0.780 Time: 0.802s RAM: 273MB
`

`
Labeled data 20%: f1_macro: 0.941, f1_micro: 0.936, roc_auc_macro: 0.990, roc_auc_micro: 0.990
Time: 0.791s RAM: 357MB
`

Sample output of **SGLA+** for clustering and embedding on Yelp:

`
Acc: 0.930 F1: 0.934 NMI: 0.739 ARI: 0.785 Time: 0.565s RAM: 274MB
`

`
Labeled data 20%: f1_macro: 0.942, f1_micro: 0.938, roc_auc_macro: 0.990, roc_auc_micro: 0.991
Time: 0.458s RAM: 359MB
`


