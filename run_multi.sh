#!/bin/bash
#SBATCH -J job_id
#SBATCH -o ./log/gere-llama-31-8b-instruct-full-multi.out
#SBATCH --gres=gpu:4 #Number of GPU devices to use [0-2]
#SBATCH --nodelist=leon09 #YOUR NODE OF PREFERENCE

module load shared apptainer 

#export CUDA_VISIBLE_DEVICES=0,1

#singularity exec --nv ./img/gere.img accelerate config
# Run raw latency/ttft
singularity exec --nv ./img/gere.img \
    bash -lc '
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    MASTER_PORT=$(shuf -i25000-30000 -n1)
    # Use python -m to avoid any PATH ambiguity with host torchrun
    /opt/env/bin/python -m torch.distributed.run \
      --nnodes=1 --nproc_per_node=4 --master_port=$MASTER_PORT \
      train_demo_multi_gpu.py
  '