#!/bin/bash
if [ -n "$1" ]
then
        temp_cuda="$1"
else
        temp_cuda="0"
fi

python3 train_coco_2017.py \
        -cf ../configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml \
        -cu $temp_cuda \
        SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025
