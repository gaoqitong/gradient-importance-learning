# Gradient Importance Learning for Incomplete Observations

Qitong Gao, Dong Wang, Joshua David Amason, Siyang Yuan, Chenyang Tao, Ricardo Henao, Majda Hadziahmetovic, Lawrence Carin, Miroslav Pajic

Paper can be found at https://openreview.net/forum?id=fXHl76nO2AZ. Accepted to ICLR '22. 

Contact: qitong.gao@duke.edu

----------------------------------------------------------------------------------------

***ATTENTION***

**Some of the data and checkpoints we uploaded require to be downloaded with Git Large File Storage, i.e., `git-lfs`.**

To install `git-lfs`, follow the instructions on https://github.com/git-lfs/git-lfs.

Once it is installed, make sure to clone this repository by running

`git lfs clone https://github.com/gaoqitong/gradient-importance-learning.git`

or

`git lfs clone git@github.com:gaoqitong/gradient-importance-learning.git`


----------------------------------------------------------------------------------------

This code package is tested against the following environmental setup:
```
Python 3.7
tensorflow 1.15.0
scikit-learn 0.24.2
pandas 1.2.4
numpy 1.20.2
scipy 1.7.0
```

Here we provided the code for training and evaluating GIL/GIL-D using multivariate tabular and sequential data. Each folder is self-contained and has a seperate readme file introducing how to train, evaluate and load pre-trained checkpoints.

If you find our work and code useful, please consider cite the paper
```
@inproceedings{
gao2022gradient,
title={Gradient Importance Learning for Incomplete Observations},
author={Qitong Gao and Dong Wang and Joshua David Amason and Siyang Yuan and Chenyang Tao and Ricardo Henao and Majda Hadziahmetovic and Lawrence Carin and Miroslav Pajic},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=fXHl76nO2AZ}
}
```
