#!/bin/bash
#SBATCH --job-name=resnet18_b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --account=csci_ga_3033_076-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --output=resnet18_A_%j.out
#SBATCH --error=resnet18_A_%j.err

singularity exec --nv \
  --overlay /scratch/VML26SP/imagenet500_train_val.sqf:ro \
  /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
  /bin/bash -c "
    source /scratch/ap9283/miniforge3/etc/profile.d/conda.sh
    conda activate cvenv
    python /scratch/ap9283/CS_3033_Vision_Meets_ML/Assignment_2/resnet18_globalnorm.py
  "
