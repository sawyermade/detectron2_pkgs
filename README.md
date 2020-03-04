# Detectron2 GitHub
FAIR holding it down hard asking Google, say what? Meet PyTorch and Detectron2 https://github.com/facebookresearch/detectron2

## Detectron2 HTTP Server GPU
```bash
# Create conda env and activate it
conda create -n detectron2 python=3.6 -y
conda activate detectron2
conda install -c pytorch pytorch torchvision -y
conda install cython -y
conda install -c menpo opencv -y
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install requests jsonpickle pyrealsense2 flask

# Clone repo and submodules
git clone --recurse-submodules https://github.com/sawyermade/detectron2_http.git

# Install detectron2 module for conda env
cd detectron2_http/detectron2
pip install -e .
```

## Detectron2 HTTP Server CPU
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