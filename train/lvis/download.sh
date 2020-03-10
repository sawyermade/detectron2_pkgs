#!/bin/bash
wget https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v0.5_train.json.zip
wget https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v0.5_val.json.zip
wget https://dl.fbaipublicfiles.com/LVIS/lvis_v0.5_image_info_test.json.zip
unzip lvis_v0.5_train.json.zip && rm -f lvis_v0.5_train.json.zip
unzip lvis_v0.5_val.json.zip && rm -f lvis_v0.5_val.json.zip
unzip lvis_v0.5_image_info_test.json.zip && rm -f lvis_v0.5_image_info_test.json.zip