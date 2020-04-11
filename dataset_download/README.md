# Download Datasets

## COCO 2017 Download
```bash
# Needs 1 argument, path to download directory.
bash coco-2017_download.sh ./coco
```

## Colab Only
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
First, make a directory on your Google Drive in 'My Drive' root named coco
```bash
# Copy to remote, remote-name is the name of the remote you created during setup
rclone copy ./coco remote-name:coco/ -vv
```
Now wait a while :)

Check Colab Notebook on training Detectron2 on COCO 2017

https://colab.research.google.com/drive/1OVStblo4Q3rz49Pe9-CJcUGkkCDLcMqP