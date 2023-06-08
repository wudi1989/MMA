# MMA
MMA: Multi-Metric-Autoencoder for Analyzing High-Dimensional and Incomplete Data

Author: Cheng Liang, Di Wu, Yi He, Teng Huang, Zhong Chen, and Xin Luo

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
## Brief Introduction
This paper proposes a Multi-Metric-Autoencoder (MMA) whose main ideas are two-fold: 1) employing different Lp-norms to build four variant Autoencoders, each of which resides in a unique metric representation space with different loss and regularization terms, and 2) aggregating these Autoencoders with a tailored, self-adaptive weighting strategy. Theoretical analysis guarantees that our MMA could attain a better representation from a set of dispersed metric spaces. Extensive experiments on four real-world datasets demonstrate that our MMA significantly outperforms seven state-of-the-art models.

## Files

The overall framework of this project is designed as follows

1. The **data_7_1_2** folder is used to hold the datasets and the testing data to calculate NDCG/Hit ;
2. The **models** is used to store the proposed model;
3. The **results** folder saves the result of the training process and the parameters of MMA;

### Enviroment Requirement
- numpy
- tensorflow (below 2.0 otherwise need to call disable_v2_behavior())

### Dataset
We offer all the dataset with 7-1-2 train-val-test ratio involved in the experiment.


### Getting Started
1. Clone this repository

```angular2html
git clone https://github.com/wudi1989/MMA.git
```

2. Make sure you meet package requirements by running:

```angular2html
pip install -r requirement.txt
```
3. Train and test MMA model

```angular2html
python main.py
```

#### Improtant arguments in main.py
- `data_name`: to choose different datasets;
    - options: "Ml1M", "Ml100k", "Hetrec-ML", "Yahoo"

- _other hyperparameters are fine-tuned and shown in main.py_

``
For example, to train and test MMA on Ml1M, just run:
``
```angular2html
python main.py --data_name "Ml1M"
```