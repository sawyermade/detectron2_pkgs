# Datasets directory
## Place symlink from coco directory to here as ./coco
```bash 
# My path was ~/DATASETS/object/COCO/2017
cd detectron2_pkgs/train
ln -s ~/DATASETS/object/COCO/2017 ./datasets/coco 
```
## Tree for coco direcotry
```
2017
├── annotations
│   └── deprecated-challenge2017
├── test2017
├── train2017
├── unlabeled2017
└── val2017
```
