#!/bin/bash
python3 train_coco_2017.py \
	-cf ../configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml \
	-dn coco_2017 \
	-tgt ~/DATASETS/object/COCO/2017/annotations/instances_train2017.json \
	-tdir ~/DATASETS/object/COCO/2017/train2017/ \
	-vgt ~/DATASETS/object/COCO/2017/annotations/instances_val2017.json \
	-vdir ~/DATASETS/object/COCO/2017/val2017/ \
	-bs 4 \
	-lr 0.0001