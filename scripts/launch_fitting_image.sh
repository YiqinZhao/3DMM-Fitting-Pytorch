#!/bin/sh

# $1: image path.
# $2: result folder.

# Load CUDA 10.1
export PATH="/mnt/SSD4T/yiqinzhao/library/cuda-10.1/cuda-toolkit/bin:$PATH"
export LD_LIBRARY_PATH="/mnt/SSD4T/yiqinzhao/library/cuda-10.1/cuda-toolkit/lib64:$LD_LIBRARY_PATH"

pants run ./third_party/face_fitting_pytorch/fit_single_img.py -- \
    --img_path=$1 \
    --res_folder=$2
