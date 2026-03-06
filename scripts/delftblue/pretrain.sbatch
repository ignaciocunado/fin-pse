#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --ntasks=1
#SBATCH --partition=gpu-a100
#SBATCH --account=Education-EEMCS-MSc-AM
#SBATCH --time=07:00:00
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=5G
#SBATCH --mail-type=BEGIN,END,FAIL

export APPTAINER_IMAGE="daic.sif"

# If you use GPUs
module load 2025
module load cuda

srun apptainer exec --nv --env-file .env -B /scratch/icunado/data $APPTAINER_IMAGE python main.py --cfg configs/ssl/pretrain.yaml