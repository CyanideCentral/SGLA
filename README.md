# SMGF: Spectrum-guided Multi-view Graph Fusion

This repository contains the implementation of **SMGF** algorithm.

## Prerequisites

Install dependencies by `conda create --n <env> --file requirements.txt -c pytorch`

Unzip the content of "data/data.zip" into "data" folder to use datasets except MAGENG,MAGPHY.

Please download complete used datasets in **url**

## Usage

**11 available datasets as follows**: 

6 muliplex datasets: ACM,DBLP,IMDB,Yelp,Freebase,RM

3 graph datasets with mulipile features: Amazon-photos,Amazon-computers,MAGENG,MAGPHY.

**2 Spectrum-guided functions for multi-view graph learning**:

SMGF-LA directly optimizes the objective with a derivative-free iterative method. 

SMGF-PI finds a surrogate objective via quadratic regression for
efficient optimization. 

Please choose the one you want to use.

Parameters used:

| Parameter     | Default | Method | Description                                                                    |
| ------------- | ------- | ------ | ------------------------------------------------------------------------------ |
| --dataset     | dblp    | LA/PI  | chooese used dataset                                                           |
| --scale       | -       | LA/PI  | configurations for large-scale data (mageng/magphy)                            |
| --knn_k       | 10      | LA/PI  | $K$, the size of neighborhood in KNN graph ,except imdb=500, yelp=200 query=20 |
| --embedding   | -       | LA/PI  | configure for generating embedding, default is clustering                      |
| --verbose     | -       | LA/PI  | produce verbose command-line output                                            |
| --embed_dim   | 64      | LA/PI  | embedding output demension                                                     |
| --embed_rank  | 32      | LA/PI  | NETMF/SKETCHNE parameter for embedding                                         |
| --eig_tol     | 0.01    | LA/PI  | precision of eigsh solver                                                      |
| --opt_t_max   | 1000    | LA/PI  | maximum number of iterations for COBYLA optimizer                              |
| --opt_epsilon | 0.01    | LA/PI  | convergence threshold for COBYLA optimizer                                     |
| --obj_alpha   | 1.0     | LA/PI  | coefficient of connectivity objective                                          |
| --obj_gamma   | 0.5     | LA/PI  | coefficient of weight regularization                                           |
| --ridge_alpha | 0.05    | PI     | regularization parameter for ridge regression                                  |

See in [command.sh](command.sh) for details
