#!/bin/bash
#SBATCH --account=project_2007839
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --time=04:20:00
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
  --d_embedding 128 \
  --d_model 512 \
  --learning_rate 0.1 \
  --dropout_rate 0.2 \
  --optimizer momo \
  --model baseline \
  --data_dir data_four_week_seq \
  --output_dir results/baseline_four_week_seq