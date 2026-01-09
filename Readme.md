# Install all in wsl anaconda

## Create conda environment
```
conda create -n adni python=3.10 -y
conda activate adni
conda install -c nvidia cuda-toolkit -y
```

Output same
```
(adni) dangnk@MSI:/mnt/c/Users/ADMIN/Desktop/sdh-dnk$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Feb_21_20:23:50_PST_2025
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0
```

### Update ubuntu if needed
```
sudo apt update
sudo apt install build-essential -y
```

### Install pytorch
```
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.4/causal_conv1d-1.5.4+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
pip install https://github.com/state-spaces/mamba/releases/download/v2.2.5/mamba_ssm-2.2.5+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```
