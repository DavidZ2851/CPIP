#!/bin/bash

# Please adjust these settings according to your needs.
#SBATCH --partition=a100_1,a100_2,v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=0406_CPIP
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jh7956@nyu.edu
#SBATCH --output=0406_CPIP.out
#SBATCH --error=0406_CPIP.err

# Singularity path
ext3_path=/scratch/$USER/overlay-25GB-500K.ext3
sif_path=/scratch/$USER/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif

cd /scratch/jh7956/CPIP
# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
python pipeline.py"