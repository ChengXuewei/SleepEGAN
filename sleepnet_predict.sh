#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1   # a GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=1-00:00:00
#SBATCH --job-name="Tiny"
#SBATCH --output=sleepnet_predict-%J.log
#SBATCH --error=sleepnet_predict-%J.error


# Print current date
date

# Load samtools
module load miniconda3
conda activate tinysleepnet

# Print name of node
# python prepare_sleepedf.py
python /rhome/xwcheng/shared_statsdept/xwcheng/tinysleepnet/predict.py --config_file config/sleepedf.py --model_dir out_sleepedf/train --output_dir out_sleepedf/predict --log_file out_sleepedf/predict.log --use-best

date

