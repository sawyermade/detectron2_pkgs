# Download Datasets

## COCO 2017 Download
```bash
# Needs 1 argument, path to download directory.
bash coco-2017_download.sh ./coco
```

## COCO Sub Directory Split for Colab
```bash
# Since in the above COCO 2017 Download we chose ./coco
python3 sub_divide_training.py ./coco
```