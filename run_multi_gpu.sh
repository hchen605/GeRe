#!/bin/bash

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5  # Use first 6 GPUs

# Distributed training parameters
NNODES=1             # Number of nodes
NPROC_PER_NODE=6     # GPUs per node
MASTER_PORT=$(shuf -i25000-30000 -n1)    # Master node port

# Launch distributed training
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_port=$MASTER_PORT \
    train_demo_multi_gpu.py