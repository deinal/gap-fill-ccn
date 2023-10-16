#!/bin/bash
#SBATCH --account=project_2007839
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=512M
#SBATCH --gres=gpu:v100:1
#SBATCH --output=slurm/%j.out


module load pytorch/1.13
pip install momo-opt

python train.py \
  --devices 1 \
  --num_workers 32 \
  --batch_size 256 \
  --epochs 60 \
  --n_head 8 \
  --n_layers 6 \
  --d_model 128 \
  --d_feedforward 256 \
  --learning_rate 0.1 \
  --dropout_rate 0.2 \
  --optimizer momo \
  --model gapt \
  --mode default \
  --data_dir data/four_week_seq \
  --output_dir results/gapt_default_four_week_seq