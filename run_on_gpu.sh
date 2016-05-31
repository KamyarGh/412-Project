#!/bin/bash
GPU=$1
echo 'Using GPU' $GPU
export CUDA_VISIBLE_DEVICES=$GPU
python "${@:2}"
