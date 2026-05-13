#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=cnn_scratch
#SBATCH --output=cnn_output_%j.out
#SBATCH --requeue

singularity exec --nv \
  /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
  /bin/bash -c "source /scratch/ap9283/miniforge3/etc/profile.d/conda.sh; 
  conda activate cvenv; 
  python /scratch/ap9283/CS_3033_Computer_Vision/Assignment_1/cnn_from_scratch.py"