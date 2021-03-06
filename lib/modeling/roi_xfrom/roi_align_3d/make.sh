#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

cd src
echo "Compiling my_lib kernels by nvcc..."
nvcc -c -o roi_align_kernel_3d.cu.o roi_align_kernel_3d.cu -x cu -Xcompiler -fPIC -arch=sm_61

cd ../
python build.py
