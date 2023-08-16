#!/bin/bash
#SBATCH --account=project_2004522
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --time=04:15:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=512M
#SBATCH --gres=gpu:v100:1
#SBATCH --output=slurm/%j.out


module load pytorch/1.13
pip install momo-opt

python train.py \
  --devices 1 \
  --num_workers 16 \
  --batch_size 128 \
  --epochs 60 \
  --n_head 8 \
  --n_layers 2 \
  --d_model 256 \
  --d_feedforward 512 \
  --learning_rate 0.1 \
  --dropout_rate 0.2 \
  --optimizer momo \
  --model gapt \
  --data_dir avg_data_env_two_weeks \
  --output_dir results/gapt_avg_data_env_two_weeks_separated