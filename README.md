# Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning


This repository is the official PyTorch implementation of the algorithms in the following paper: 

Cong Zhang, Wen Song, Zhiguang Cao, Jie Zhang, Puay Siew Tan, Chi Xu. Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning. 34th Conference on Neural Information Processing Systems (NeurIPS), 2020. [\[PDF\]](https://proceedings.neurips.cc/paper/2020/file/11958dfee29b6709f48a9ba0387a2431-Paper.pdf)


If you make use of the code/experiment or L2D algorithm in your work, please cite our paper (Bibtex below).
```
@inproceedings{NEURIPS2020_11958dfe,
 author = {Zhang, Cong and Song, Wen and Cao, Zhiguang and Zhang, Jie and Tan, Puay Siew and Chi, Xu},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {1621--1632},
 publisher = {Curran Associates, Inc.},
 title = {Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning},
 url = {https://proceedings.neurips.cc/paper/2020/file/11958dfee29b6709f48a9ba0387a2431-Paper.pdf},
 volume = {33},
 year = {2020}
}
```

## Installation
Pytorch 1.6

Gym 0.17.3

### Docker install
Clone this repo and within the repo folder run the following command.

Create image `l2d-image`:
```commandline
sudo docker build -t l2d-image .
```

Create container `l2d-container` from `l2d-image`, and activate it:
```commandline
sudo docker run --gpus all --name l2d-container -it l2d-image
```

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
