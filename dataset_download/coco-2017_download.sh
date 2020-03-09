#!/bin/bash

# Download directory
outdir="$1"

# Download Images
wget http://images.cocodataset.org/zips/train2017.zip -P $outdir -N
wget http://images.cocodataset.org/zips/val2017.zip -P $outdir -N
wget http://images.cocodataset.org/zips/test2017.zip -P $outdir -N
wget http://images.cocodataset.org/zips/unlabeled2017.zip -P $outdir -N

# Download Annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P $outdir -N
wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip -P $outdir -N
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip -P $outdir -N
wget http://images.cocodataset.org/annotations/image_info_test2017.zip -P $outdir -N
wget http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip -P $outdir -N