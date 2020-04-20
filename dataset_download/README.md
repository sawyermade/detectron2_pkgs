# Download Datasets

## COCO 2017 Download
```bash
# Needs 1 argument, path to download directory.
bash coco-2017_download.sh ./coco
```

## Colab Setup Only
### COCO Sub Directory Split for Colab
```bash
# Since in the above COCO 2017 Download we chose ./coco
python3 sub_divide_training.py ./coco 

# Remove old train and test so you dont have to upload them too
rm -rf ./coco/train2017 ./coco/test2017
```

### Upload to Google Drive
#### Install rclone
```bash
curl https://rclone.org/install.sh | sudo bash
```

#### Link rclone with google drive, figure it out yourself I believe in you :)
https://rclone.org/drive/

#### Upload to Google Drive
First, make a directory on your Google Drive in 'My Drive' root named COCO
```bash
# Copy to remote, remote-name is the name of the remote you created during setup
rclone copy ./coco remote-name:COCO/ -vv

# rclone copy val2017_sub-500 gdrive-usf:DATASETS/COCO/2017/val2017_sub-500 -vv --drive-chunk-size=256M --transfers=40 --checkers=40 --tpslimit=9 --fast-list --max-backlog 200000
```
Now wait a while :)

### Colab Notebook on Training Detectron2 with COCO 2017
https://colab.research.google.com/drive/1OVStblo4Q3rz49Pe9-CJcUGkkCDLcMqP
