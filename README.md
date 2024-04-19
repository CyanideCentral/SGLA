# SMGF: Spectrum-guided Multi-view Graph Fusion

This repository contains the implementation of **SMGF** algorithm.

## Prerequisites

Install dependencies by `conda create --name <env> --file requirements.txt -c pytorch`.

Unzip the content of "data.zip" into "data" folder by `unzip data.zip` to use datasets except MAG-eng & MAG-phy.

Please download complete used datasets including MAG-eng & MAG-phy in **url**.

## Usage

**11 available datasets as follows**: 

6 muliplex datasets: ACM, DBLP, IMDB, Yelp, Freebase, RM.

5 graph datasets with mulipile features: Query, Amazon-photos, Amazon-computers, MAG-eng, MAG-phy.

**2 Spectrum-guided functions for multi-view graph learning**:

SMGF directly optimizes the objective with a derivative-free iterative method. 

SMGFQ finds a surrogate objective via quadratic regression for efficient optimization. 

Please choose the one you want to use.

Parameters used:

| Parameter     | Default | Description                                                  |
| ------------- | ------- | ------------------------------------------------------------ |
| --dataset     | dblp    | chooese used dataset                                         |
| --scale       | -       | configurations for large-scale data (mageng/magphy)          |
| --verbose     | -       | produce verbose command-line output                          |
| --embedding   | -       | configure for generating embedding, default clustering       |
| --embed_dim   | 64      | embedding output demension                                   |
| --embed_rank  | 32      | NETMF/SKETCHNE parameter for embedding                       |
| --eig_tol     | 0.01    | precision of eigsh solver                                    |
| --knn_k       | 10      | $K$, k neighbors except imdb=500, yelp=200, query=20         |
| --opt_t_max   | 100     | $T_{max}$, maximum number of iterations for COBYLA optimizer |
| --opt_epsilon | 0.01    | $\epsilon$, convergence threshold for COBYLA optimizer       |
| --obj_alpha   | 1.0     | $\alpha$, coefficient of connectivity objective              |
| --obj_gamma   | 0.5     | $\gamma$, coefficient of weight regularization               |
| --ridge_alpha | 0.05    | $a_r$, regularization parameter for ridge regression         |

See [command.sh](command.sh) for command line in details
