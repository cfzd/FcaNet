export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NGPUS=8
export OMP_NUM_THREADS=12

# Before testing, please edit the `pretrained` path of pretrained ImageNet FcaNet model in \
# ./mmdetection/configs/_base_/models/faster_rcnn_r50_fpn_freqnet.py
# ./mmdetection/configs/_base_/models/faster_rcnn_r101_fpn_freqnet.py
# ./mmdetection/configs/_base_/models/mask_rcnn_r50_fpn_freqnet.py

# And modify the `data_root` path of COCO in \
# ./mmdetection/configs/_base_/datasets/coco_detection.py
# ./mmdetection/configs/_base_/datasets/coco_instance.py

# To eval the results of detection with FcaNet50 and FcaNet101, please run
./mmdetection/tools/dist_test.sh 'mmdetection/configs/faster_rcnn/faster_rcnn_freqnet50_fpn_1x_coco.py' 'fcanet_faster_rcnn50.pth' $NGPUS --eval bbox
./mmdetection/tools/dist_test.sh 'mmdetection/configs/faster_rcnn/faster_rcnn_freqnet101_fpn_1x_coco.py' 'fcanet_faster_rcnn101.pth' $NGPUS --eval bbox

# To eval the results of instance segmentation with FcaNet50, please run
./mmdetection/tools/dist_test.sh 'mmdetection/configs/mask_rcnn/mask_rcnn_freqnet50_fpn_1x_coco.py' 'fcanet_mask_rcnn50.pth' $NGPUS --eval bbox segm