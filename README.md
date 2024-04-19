# SMGF: Spectrum-guided Multi-view Attributed Graph Fusion

This repository contains the implementation of **SMGF**  and **SMGFQ** algorithms.

## Prerequisites

Install dependencies via Conda:
```
conda create --name SMGF --file requirements.txt -c pytorch
conda activate SMGF
```

Extract dataset files: 
```
unzip data.zip
```

## Usage

The following command runs our algorithm on a certain dataset, with configurations as specified below.

```
python <ALGORITHM>.py --dataset <DATASET> <OPTIONS>
```

### Algorithms

We provide two algorithms for multi-view attributed graph clustering and embedding: (see our paper for details)

- **SMGF** iteratively minimizes the objective with the derivative-free COBYLA optimizer. 

- **SMGFQ** uses a quadratic interpolation technique to find an approximate optimum. 

### Datasets

Replace \<DATASET\> with any of the following dataset names:

ACM, DBLP, IMDB, Yelp, Freebase, RM, Query, Amazon-photos, Amazon-computers.

Given the size limit on GitHub repositories, the MAG-phy and MAG-eng datasets will be released later.

### Tasks

By default, the above command runs the clustering task. For the embedding task, add `--embedding` to the command line arguments.

### Full command line options

| Parameter     | Default | Description                                                  |
| ------------- | ------- | ------------------------------------------------------------ |
| --dataset     | dblp    | Selected dataset                                         |
| --embedding   | -       | Execute the embedding task instead of clustering       |
| --knn_k       | 10      | $K$ in attribute KNNs (200 for Yelp, 500 for IMDB, 20 for Query)         |
| --eig_tol     | 0.01    | Precision for EIGSH eigensolver                                    |
| --opt_t_max   | 100     | $T_{max}$, maximum number of iterations for COBYLA optimizer |
| --opt_epsilon | 0.01    | $\epsilon$, convergence threshold for COBYLA optimizer       |
| --obj_alpha   | 1.0     | $\alpha$, coefficient of connectivity objective              |
| --obj_gamma   | 0.5     | $\gamma$, coefficient of weight regularization               |
| --ridge_alpha | 0.05    | $a_r$, regularization parameter for ridge regression (SMGFQ only)        |
| --embed_dim   | 64      | Dimension of node embeddings                                  |
| --embed_rank  | 32      | NETMF/SKETCHNE embedding algorithm parameter (64 for Freebase)                 |
| --scale       | -       | Enable scalability configurations for MAG-eng and MAG-phy datasets         |
| --verbose     | -       | Show verbose command-line output                          |

### Examples

The following commands reproduce our results on the Yelp dataset.

#### **SMGF** clustering
```
python SMGF.py --dataset yelp --knn_k 200
```
Sample output: 
```
Acc: 0.930 F1: 0.934 NMI: 0.739 ARI: 0.785 Time: 0.739s RAM: 214MB
```
#### **SMGF** embedding
```
python SMGF.py --dataset yelp --knn_k 200 --embedding
```
Sample output: 
```
Labeled data 20%: f1_macro: 0.943, f1_micro: 0.938, roc_auc_macro: 0.990, roc_auc_micro: 0.991
Time: 2.379s RAM: 421MB
```

#### **SMGFQ** clustering
```
python SMGFQ.py --dataset yelp --knn_k 200
```
Sample output: 
```
Acc: 0.930 F1: 0.932 NMI: 0.733 ARI: 0.787 Time: 1.178s RAM: 215MB
```
#### **SMGFQ** embedding
```
python SMGFQ.py --dataset yelp --knn_k 200 --embedding
```
Sample output: 
```
Labeled data 20%: f1_macro: 0.942, f1_micro: 0.937, roc_auc_macro: 0.990, roc_auc_micro: 0.991
Time: 1.823s RAM: 419MB
```
