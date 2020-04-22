# Detectron2 Quick N' Dirty HTTP Server 
## Run server
```bash 
# Go into cloned directory and activate conda env
cd detectron2_pkgs/http
conda activate detectron2

# Run server
bash run_server_defaults.sh
```

## Run client demo
```bash
# Enter directory and activate environment
cd detectron2_pkgs/http
conda activate detectron2

# Run with realsense
bash run_client_rs.sh

# Run with OpenCV webcam capture, default capture device 0
bash run_client_wc.sh
```
