# Detectron2 GitHub
FAIR holding it down hard asking Google, say what? Meet PyTorch and Detectron2 https://github.com/facebookresearch/detectron2

# Detectron2 HTTP server
```bash
# Create conda env and activate it
conda create -n detectron2 python=3.6 pytorch cython numpy -y
conda activate detectron2
conda install -c menpo opencv -y
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install requests jsonpickle pyrealsense2 flask

# Clone repo and submodules
git clone --recurse-submodules https://github.com/sawyermade/detectron2_http.git

# Install detectron2 module for conda env
cd detectron2_http/detectron2
pip install -e .
```
