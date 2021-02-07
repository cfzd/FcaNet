export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NGPUS=8
export OMP_NUM_THREADS=12

# Before training, please edit the `pretrained` path of pretrained ImageNet FcaNet model in \
# ./mmdetection/configs/_base_/models/faster_rcnn_r50_fpn_freqnet.py
# ./mmdetection/configs/_base_/models/faster_rcnn_r101_fpn_freqnet.py
# ./mmdetection/configs/_base_/models/mask_rcnn_r50_fpn_freqnet.py

# And modify the `data_root` path of COCO in \
# ./mmdetection/configs/_base_/datasets/coco_detection.py
# ./mmdetection/configs/_base_/datasets/coco_instance.py


# To train a Faster RCNN using FcaNet50, please run
./mmdetection/tools/dist_train.sh 'mmdetection/configs/faster_rcnn/faster_rcnn_freqnet50_fpn_1x_coco.py' $NGPUS --work-dir path/to/save/result

# To train a Faster RCNN using FcaNet101, please run
./mmdetection/tools/dist_train.sh 'mmdetection/configs/faster_rcnn/faster_rcnn_freqnet101_fpn_1x_coco.py' $NGPUS --work-dir path/to/save/result

# To train a Mask RCNN using FcaNet50, please run
./mmdetection/tools/dist_train.sh 'mmdetection/configs/mask_rcnn/mask_rcnn_freqnet50_fpn_1x_coco.py' $NGPUS --work-dir path/to/save/result