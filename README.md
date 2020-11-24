# Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning

## This is a temporary README.md, will update soon.

This repository is the official PyTorch implementation of the algorithms in the following paper: 

Cong Zhang, Wen Song, Zhiguang Cao, Jie Zhang, Puay Siew Tan, Chi Xu. Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning. 34th Conference on Neural Information Processing Systems (NeurIPS), 2020. [\[PDF\]](https://proceedings.neurips.cc/paper/2020/file/11958dfee29b6709f48a9ba0387a2431-Paper.pdf)


If you make use of the code/experiment or L2D algorithm in your work, please cite our paper (Bibtex below).
```
@article{zhang2020learning,
  title={Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning},
  author={Zhang, Cong and Song, Wen and Cao, Zhiguang and Zhang, Jie and Tan, Puay Siew and Chi, Xu},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

## Installation
Pytorch 1.6

Gym 0.17.3

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
