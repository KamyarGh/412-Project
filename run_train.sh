#!/bin/bash
# GPU=$(gpu_lock2.py --id)
GPU=0
echo 'Using GPU' $GPU
export CUDA_VISIBLE_DEVICES=$GPU
python train.py $1
