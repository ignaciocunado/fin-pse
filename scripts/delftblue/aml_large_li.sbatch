#!/bin/bash
#SBATCH --job-name=aml_large_li
#SBATCH --ntasks=1
#SBATCH --partition=gpu-a100
#SBATCH --account=Education-EEMCS-MSc-AM
#SBATCH --time=15:00:00
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=7G
#SBATCH --mail-type=BEGIN,END,FAIL

export APPTAINER_IMAGE="daic.sif"

# If you use GPUs
module load 2025
module load cuda

srun apptainer exec --nv --env-file .env -B /scratch/icunado/data $APPTAINER_IMAGE python main.py --cfg configs/aml/aml_large_li.yaml --repeat 5