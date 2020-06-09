#!/bin/bash
temp_cuda="0"
if [ -n "$1" ]
then
        temp_cuda="$1"
fi

temp_config="../configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
if [ -n "$2" ]
then 
	temp_config="$2"
fi

#echo "cuda: $temp_cuda"
#echo "config: $temp_config"

python3 train_coco_2017.py \
        -cf $temp_config \
        -cu $temp_cuda \
        SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
