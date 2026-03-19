#!/bin/bash
#SBATCH --job-name=explore
#SBATCH --nodes=1
#SBATCH --account=csci_ga_3033_076-2026sp
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --mem=8G
#SBATCH --output=explore.out
#SBATCH --error=explore.err

singularity exec --nv \
  --overlay /scratch/VML26SP/imagenet500-train.sqf:ro \
  --overlay /scratch/VML26SP/imagenet500-val.sqf:ro \
  /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
  /bin/bash -c "
    echo '=== ROOT ===' && ls / && \
    echo '=== LOOKING FOR JPEGS ===' && \
    find / -name '*.JPEG' -maxdepth 6 2>/dev/null | head -10
  "