#!/bin/sh
#SBATCH --job-name="aml_small_li"
#SBATCH --partition=ewi-st,general
#SBATCH --qos=medium
#SBATCH --time=07:00:00
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

export APPTAINER_IMAGE="daic.sif"

# If you use GPUs
module use /opt/insy/modulefiles
module load cuda/11.8

# Run script

srun apptainer exec --nv --env-file .env -B /tudelft.net/staff-umbrella/CSE3000GLTD/ignacio/fin-pse $APPTAINER_IMAGE python main.py --cfg configs/aml/aml_small_li.yaml