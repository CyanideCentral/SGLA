# SGLA and SGLA+: Efficient Integration of Multi-View Attributed Graphs for Clustering and Embedding

This repository contains the implementation of **SGLA**  and **SGLA+** algorithms for multi-view attributed graphs. The research paper will be presented at ICDE 2025.

## Prerequisites

Install dependencies:
```
conda create --name SGLA --file requirements.txt -c pytorch
```

6 MVAG datasets (RM, Yelp, IMDB, DBLP, Amazon photos, Amazon computers) can be extracted from the zipfile.
```
unzip data.zip
```

(Optional) Large-scale datasets MAG-eng and MAG-phy are available via [Zenodo](https://zenodo.org/records/15099668). Extract dataset files by:
```
unzip <download_path>/mag_data.zip -d data/
```

## Usage

Two algorithms for the integration of multi-view attributed graphs (MVAG) are provided:

**SGLA** optimizes the objective with a derivative-free iterative method in **SGLA.py**.

**SGLA+** adopts sampling and quadratic interpolation to boost efficiency in **SGLA_plus.py**.

Available command line options:
| Parameter     | Default | Description                                                  |
| ------------- | ------- | ------------------------------------------------------------ |
| --dataset     | dblp    | Name of the multi-view attributed graph dataset                                        |
| --scale       | -       | Scalable mode (required for MAG-eng and MAG-phy datasets)          |
| --verbose     | -       | Produce verbose command-line output                          |
| --embedding   | -       | Run embedding task (The default task is clustering)       |
| --embed_dim   | 64      | Dimension of generated node embeddings                                   |
| --knn       | 10      | $K$ for KNN (Set 200 for Yelp, 500 for IMDB)         |
| --tmax   | 50     | $T_{max}$, maximum number of iterations for COBYLA optimizer |
| --epsilon | 0.001   | $\epsilon$, convergence threshold for COBYLA optimizer       |
| --gamma   | 0.5     | $\gamma$, regularization hyperparameter               |
| --ridge_alpha | 0.05    | $a_r$, regularization parameter for ridge regression         |

To **reproduce** the results in our paper, use following commands to run SGLA.py (or SGLA_plus.py for SGLA+) for clustering or append " --embedding" for embedding.

```
python SGLA.py --dataset rm
python SGLA.py --dataset yelp --knn 200
python SGLA.py --dataset idmb --knn 500
python SGLA.py --dataset dblp
python SGLA.py --dataset amazon-photos
python SGLA.py --dataset amazon-computers
python SGLA.py --dataset mageng --scale
python SGLA.py --dataset magphy --scale
```

## Examples

We give experiment details on the Yelp dataset for reference.

#### **SGLA**
##### Run clustering on Yelp
```
python SGLA.py --dataset yelp --knn 200
```
##### Run emebedding on Yelp
```
python SGLA.py --dataset yelp --knn 200 --embedding
```
##### Sample output
Clustering:
`
Acc: 0.927 F1: 0.930 NMI: 0.728 ARI: 0.780 Time: 0.802s RAM: 273MB
`

Embedding:
`
Labeled data 20%: f1_macro: 0.941, f1_micro: 0.936, roc_auc_macro: 0.990, roc_auc_micro: 0.990
Time: 0.791s RAM: 357MB
`

#### **SGLA+**
##### Run clustering on Yelp
```
python SGLA_plus.py --dataset yelp --knn 200
```
##### Run emebedding on Yelp
```
python SGLA_plus.py --dataset yelp --knn 200 --embedding
```

##### Sample output
Clustering:
`
Acc: 0.930 F1: 0.934 NMI: 0.739 ARI: 0.785 Time: 0.565s RAM: 274MB
`

Embedding:
`
Labeled data 20%: f1_macro: 0.942, f1_micro: 0.938, roc_auc_macro: 0.990, roc_auc_micro: 0.991
Time: 0.458s RAM: 359MB
`
