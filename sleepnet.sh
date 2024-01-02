#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1   # a GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=10-00:00:00
#SBATCH --job-name=tiny
#SBATCH --output=sleepnet-%J.log
#SBATCH --error=sleepnet-%J.error


# Print current date
date

# Load samtools
module load miniconda3
conda activate tinysleepnet

# Print name of node
# python prepare_sleepedf.py
python /rhome/xwcheng/shared_statsdept/xwcheng/tinysleepnet/trainer.py --db sleepedf --gpu 0 --from_fold 0 --to_fold 19

date
