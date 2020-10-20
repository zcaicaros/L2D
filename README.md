# Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning

## This is a temporary README.md, will update soon.

This repository is the official PyTorch implementation of the algorithms in the following paper: 

Cong Zhang, Wen Song, Zhiguang Cao, Jie Zhang, Puay Siew Tan, Chi Xu. Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning. 34th Conference on Neural Information Processing Systems (NeurIPS), 2020. [paper](https://github.com/zcajiayin/L2D/blob/main/paper/ID7527_camra_ready-merged.pdf)


If you make use of the code/experiment or L2D algorithm in your work, please cite our paper (Bibtex below).
```
@incollection{NIPS2020_7527,
title = {Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning},
author = {Zhang, Cong and Song, Wen and Cao, Zhiguang and Zhang, Jie and Tan, Puay Siew and Xu, Chi},
booktitle = {Advances in Neural Information Processing Systems 33},
year = {2020},
}
```

## Installation
Pytorch 1.6

## Reproduce result in paper
Change the device type in ```Params.py``` file and run:
```
python3 test_learned.py
```

### Or
Change the device type in ```Params.py``` file and run:
```
python3 test_learned_on_benchmark.py
```
for open benchmark
