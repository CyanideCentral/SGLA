# SMGF: Spectrum-guided Multi-view Graph Fusion

This repository contains the implementation of **SMGF** algorithm.

## Prerequisites

Install dependencies by `conda create --name <env> --file requirements.txt -c pytorch`.

Unzip the content of "data.zip" into "data" folder by `unzip data.zip` to use datasets except MAGENG & MAGPHY.

Please download complete used datasets including MAGENG & MAGPHY in **url**.

## Usage

**11 available datasets as follows**: 

6 muliplex datasets: ACM, DBLP, IMDB, Yelp, Freebase, RM.

5 graph datasets with mulipile features: Query, Amazon-photos, Amazon-computers, MAGENG, MAGPHY.

**2 Spectrum-guided functions for multi-view graph learning**:

SMGF-LA directly optimizes the objective with a derivative-free iterative method. 

SMGF-PI finds a surrogate objective via quadratic regression for efficient optimization. 

Please choose the one you want to use.

Parameters used:

| Parameter     | Default | Method | Description                                            |
| ------------- | ------- | ------ | ------------------------------------------------------ |
| --dataset     | dblp    | LA/PI  | chooese used dataset                                   |
| --scale       | -       | LA/PI  | configurations for large-scale data (mageng/magphy)    |
| --knn_k       | 10      | LA/PI  | $K$, k neighbors except imdb=500, yelp=200, query=20   |
| --embedding   | -       | LA/PI  | configure for generating embedding, default clustering |
| --verbose     | -       | LA/PI  | produce verbose command-line output                    |
| --embed_dim   | 64      | LA/PI  | embedding output demension                             |
| --embed_rank  | 32      | LA/PI  | NETMF/SKETCHNE parameter for embedding                 |
| --eig_tol     | 0.01    | LA/PI  | precision of eigsh solver                              |
| --opt_t_max   | 1000    | LA/PI  | $T_{max}$, maximum number of iterations for COBYLA optimizer      |
| --opt_epsilon | 0.01    | LA/PI  | $\epsilon$, convergence threshold for COBYLA optimizer             |
| --obj_alpha   | 1.0     | LA/PI  | $\alpha$, coefficient of connectivity objective                  |
| --obj_gamma   | 0.5     | LA/PI  | $\gamma$, coefficient of weight regularization                   |
| --ridge_alpha | 0.05    | PI     | $a_r$, regularization parameter for ridge regression          |

See [command.sh](command.sh) for command line in details
