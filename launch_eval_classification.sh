export CUDA_VISIBLE_DEVICES=0,1,2,3
export NGPUS=4
export OMP_NUM_THREADS=12

# To eval FcaNet 34, 50, 101 and 152, please run
python -m torch.distributed.launch --nproc_per_node=$NGPUS main.py -a fcanet34 --dali_cpu --b 128 -e --evaluate_model /path/to/your/fca34.pth /path/to/your/ImageNet
python -m torch.distributed.launch --nproc_per_node=$NGPUS main.py -a fcanet50 --dali_cpu --b 128 -e --evaluate_model /path/to/your/fca50.pth /path/to/your/ImageNet
python -m torch.distributed.launch --nproc_per_node=$NGPUS main.py -a fcanet101 --dali_cpu --b 128 -e --evaluate_model /path/to/your/fca101.pth /path/to/your/ImageNet
python -m torch.distributed.launch --nproc_per_node=$NGPUS main.py -a fcanet152 --dali_cpu --b 128 -e --evaluate_model /path/to/your/fca152.pth /path/to/your/ImageNet

# To eval the experiments about DCT/Learned tensor, please run
python -m torch.distributed.launch --nproc_per_node=$NGPUS main.py -a fcanet50 --dali_cpu --b 128 -e --evaluate_model /path/to/your/fixed_rand.pth /path/to/your/ImageNet
python -m torch.distributed.launch --nproc_per_node=$NGPUS main.py -a fcanet50 --dali_cpu --b 128 -e --evaluate_model /path/to/your/learn_rand.pth /path/to/your/ImageNet
python -m torch.distributed.launch --nproc_per_node=$NGPUS main.py -a fcanet50 --dali_cpu --b 128 -e --evaluate_model /path/to/your/learn_dct.pth /path/to/your/ImageNet
