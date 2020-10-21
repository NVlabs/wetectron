## Installation

### Requirements:
- Python 3
- PyTorch 1.5+ (Installation instructions: https://pytorch.org/get-started/locally/)
- torchvision

### Getting started

```bash
# From a clean conda env, first do
conda create --name wetectron python=3.7
conda activate wetectron

# wetectron dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python tensorboardX pycocotools

# PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 10.1
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# install wetectron
git clone https://github.com/nvlabs/wetectron/
cd wetectron

# install apex
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../

# the following will install the lib with symbolic links, so that you can modify
# the files if you want and won't need to re-build it
# To builde for multiple GPU arch, use `export TORCH_CUDA_ARCH_LIST="3.7;5.0;6.0;7.0"`
python setup.py build develop
```