# Detectron2 GitHub
FAIR Detectron2 https://github.com/facebookresearch/detectron2

## Server Setup and Install
### Detectron2 Conda Environment Setup GPU
```bash
# May not work if driver and cuda version arent the same as mine: Driver 440 and CUDA 10.2, try from scratch below
git clone --recurse-submodules https://github.com/sawyermade/detectron2_pkgs.git
cd detectron2_pkgs
conda env create -f conda_env.yaml
conda activate detectron2
cd detectron2
pip install -e .
pip install git+https://github.com/lvis-dataset/lvis-api.git
```

### Detectron2 HTTP Server GPU From Scratch
```bash
# Create conda env and activate it
conda create -n detectron2 python=3.6 -y
conda activate detectron2
conda install -c pytorch pytorch torchvision -y
conda install cython imageio -y
conda install -c menpo opencv3 -y
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install requests jsonpickle pyrealsense2 flask

# Clone repo and submodules
git clone --recurse-submodules https://github.com/sawyermade/detectron2_pkgs.git
cd detectron2_pkgs

# Install detectron2 module for conda env
cd detectron2
pip install -e .
```

### Detectron2 HTTP Server CPU
```bash
conda create -n detectron2cpu python=3.6 -y
conda activate detectron2cpu
pip install detectron2 -f \
	https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/index.html
conda install pytorch torchvision cpuonly -c pytorch
conda install cython -y
conda install -c menpo opencv -y
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install requests jsonpickle pyrealsense2 flask
```

## Client Demo
```bash
# Enter directory and activate environment
cd detectron2_pkgs/http
conda activate detectron2

# Run with realsense
./run_client_rs.sh

# Run with OpenCV webcam capture
./run_client_wc.sh
```