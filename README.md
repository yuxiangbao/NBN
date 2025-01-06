# Normalizing Batch Normalization for Long-Tailed Recognition

<div align=center>
<img src="assets/motivation.png" style="width:70%;">
</div>


## 🛠️ Installation
```shell
pip install -r requirements.txt
```

## 🚀 Training and Validation

### 1. Training Data Preparation

Before training, please download the datasets following [Kang](https://github.com/facebookresearch/classifier-balancing). Then, update the `data_root` parameter in the YAML configuration files found in the `./config` directory.

### 2. Training

We provide the [launch.sh](./launch.sh) script to initiate the training process. This script supports both single-GPU and multi-GPU training using FP32 and FP16 precision modes. For multi-GPU setups, it leverages Data Parallel and Distributed Data Parallel (DDP) for efficient parallel processing.
```shell
# training with single gpu
python main.py config/ImageNet_LT/ride.yaml --gpu $gpu_id --amp

# training with data parallel
python main.py config/ImageNet_LT/ride.yaml --amp

# training with distributed data parallel
python main.py config/ImageNet_LT/ride.yaml -d --world-size 1 --rank 0 --amp
```

### 3. Validation

Likewise, the code also support single-gpu and multi-gpu validation. For multi-GPU validation, we use Distributed Data Parallel (DDP) as an example.

```shell
python main.py config/ImageNet_LT/ride_nbn_lr.yaml -d --world-size 1 --rank 0 --amp -e --pretrain /path/to/checkpoints
```

The checkpoints can be freely downloaded from [Google Drive](https://drive.google.com/file/d/1ebEgsvbp00AvKcfa2JQRK8rVoQ6Ab38C/view?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1feTkJ6h3HUG24lehOr9BUw?pwd=6fdv).


## ⭐ Cite

If you find this project useful in your research, we appreciate your citation of our work:

```
@article{bao2024normalizing,
  title={Normalizing Batch Normalization for Long-Tailed Recognition},
  author={Bao, Yuxiang and Kang, Guoliang and Yang, Linlin and Duan, Xiaoyue and Zhao, Bo and Zhang, Baochang},
  journal={IEEE Transactions on Image Processing},
  year={2024},
  publisher={IEEE}
}
```

## 🎖️ Acknowledgement
This work is built upon the [decoupling cRT](https://github.com/facebookresearch/classifier-balancing), [Balanced Softmax](https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification), and [RoBal](https://github.com/wutong16/Adversarial_Long-Tail).

## 🦄 Contact
Please contact [@yuxiangbao](https://github.com/yuxiangbao) for questions, comments and reporting bugs.