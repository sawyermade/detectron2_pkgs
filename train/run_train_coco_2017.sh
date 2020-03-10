#!/bin/bash
python3 train_coco_2017.py \
	-cf ../configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml \
	SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025