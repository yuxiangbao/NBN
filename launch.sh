# training with single gpu
python main.py config/ImageNet_LT/ride.yaml --gpu $gpu_id --amp

# training with data parallel
python main.py config/ImageNet_LT/ride.yaml --amp

# training with distributed data parallel
python main.py config/ImageNet_LT/ride.yaml -d --world-size 1 --rank 0 --amp

# Likewise, you can also validate with single gpu, data parallel or distributed data parallel. 
# Here we take distributed data parallel for example.
python main.py config/ImageNet_LT/ride_nbn_lr.yaml -d --world-size 1 --rank 0 --amp \
-e --pretrain /path/to/checkpoints