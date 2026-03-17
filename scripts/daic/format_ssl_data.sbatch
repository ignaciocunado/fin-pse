#!/bin/sh
#SBATCH --job-name="format_ssl_data"
#SBATCH --partition=ewi-st,general
#SBATCH --qos=short
#SBATCH --time=00:16:00
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:0
#SBATCH --mem=512GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

export APPTAINER_IMAGE="daic.sif"
# Run script

srun apptainer exec --nv --env-file .env -B /tudelft.net/staff-umbrella/CSE3000GLTD/ignacio/fin-pse $APPTAINER_IMAGE python etl/aml/transform_cumulative_window_convert_currencies.py
srun apptainer exec --nv --env-file .env -B /tudelft.net/staff-umbrella/CSE3000GLTD/ignacio/fin-pse $APPTAINER_IMAGE python etl/aml/transform_whole_dataset_convert_currencies.py
srun apptainer exec --nv --env-file .env -B /tudelft.net/staff-umbrella/CSE3000GLTD/ignacio/fin-pse $APPTAINER_IMAGE python etl/aml/transform_window.py
srun apptainer exec --nv --env-file .env -B /tudelft.net/staff-umbrella/CSE3000GLTD/ignacio/fin-pse $APPTAINER_IMAGE python etl/aml/transform_window_convert_currencies.py

