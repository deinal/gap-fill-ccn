#!/bin/bash
#SBATCH --account=project_2004522
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:v100:1
#SBATCH --output=slurm/%j.out


module load pytorch/1.13

python train.py \
  --devices 1 \
  --num_workers 16 \
  --batch_size 32 \
  --epochs 100 \
  --n_head 1 \
  --n_layers 2 \
  --data_dir avg_data \
  --output_dir results/avg-100-nhead-1-target