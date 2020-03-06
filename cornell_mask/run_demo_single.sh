#!/bin/bash
python3 get_cornell_masks.py \
	--input-dir outputs \
	--output outputs_masks_crop-${1}/ \
	--confidence-threshold 0.${1} \
	--config-file configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml \
	--cornell-single