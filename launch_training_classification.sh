export CUDA_VISIBLE_DEVICES=0,1,2,3
export NGPUS=4
export OMP_NUM_THREADS=12

# For example, to train a FcaNet50 without mixed precision, please run
python -m torch.distributed.launch --nproc_per_node=$NGPUS main.py -a fcanet50 --dali_cpu --b 64 /path/to/your/ImageNet

# To train a FcaNet50 using APEX mixed precision, please run
python -m torch.distributed.launch --nproc_per_node=$NGPUS main.py -a fcanet50 --dali_cpu --b 128 --loss-scale 128.0 --opt-level O2 /path/to/your/ImageNet

# The training for FcaNet34, FcaNet101, and FcaNet152 is similar, but you might need to adjust the batch size according to your CUDA mem.
# The learning rate should NOT be adjusted, since the linear scaling rule has been implemented and it will automatically change lr based on your batch size.

