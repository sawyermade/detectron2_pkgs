# Mask Results Cornell Grasping Dataset

## Detectron 1
```
Need to rerun for exact numbers but was somewhere around 250/885
```

## Detectron 2
### Not Cropped
```
Have to devise way to get exact count what worked, remove refridgerator and dinning table?
```

### Cropped
```
Mask RCNN COCO
1. mask_rcnn_R_50_FPN_inference_acc_test.yaml @ 0.50: 381/885

2. mask_rcnn_R_50_FPN_inference_acc_test.yaml @ 0.40: 445/885

3. mask_rcnn_R_50_FPN_inference_acc_test.yaml @ 0.30: 510/885

4. mask_rcnn_R_50_FPN_inference_acc_test.yaml @ 0.25: 557/885

5. mask_rcnn_R_50_FPN_inference_acc_test.yaml @ 0.20: 605/885

LIVS 
1. mask_rcnn_X_101_32x8d_FPN_1x.yaml @ 0.50: 0/885

Panoptic COCO
1. panoptic_fpn_R_101_3x.yaml @ 0.50: 0/885
```
