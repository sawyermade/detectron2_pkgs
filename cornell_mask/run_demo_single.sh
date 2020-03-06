#!/bin/bash
python3 get_cornell_masks.py \
	--input-dir outputs \
	--output outputs_masks_crop-${2}/ \
	--confidence-threshold 0.${2} \
	--config-file $1 \
	--cornell-single