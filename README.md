# SMGF: Spectrum-guided Multi-view Graph Fusion

This repository contains the implementation of **SMGF** algorithm.

## Prerequisites

Install dependencies by `pip install -r requirements.txt`

Unzip the content of "data/data.zip" into "data" folder. 

## Usage

**10 available datasets as follows**: 

6 muliplex datasets: acm,dblp,imdb,yelp,freebase,rm

4 graph datasets with mulipile features: amazon-photos,amazon-computers,magphy,mageng.

**2 Spectrum-guided functions for multi-view graph learning**:

SMGF-LA directly optimizes the objective with a derivative-free iterative method. Alternatively, SMGF-PI finds a surrogate objective via quadratic regression for
efficient optimization. Please choose the one you want to use.

Parameters used:

| Parameter   | Default | Description                                             |
| ----------- | ------- | ------------------------------------------------------- |
| --dataset   | dblp    | chooese used dataset                                    |
| --knn_k     | 10      | $K$, the size of neighborhood in KNN graph              |
| --embedding | -       | configure for generate embedding, default is clustering |
| --verbose   | -       | Produce verbose command-line output                     |

See in [command.sh](command.sh) for details
