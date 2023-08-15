#!/bin/bash
#SBATCH --account=project_2004522
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
  --epochs 120 \
  --n_head 1 \
  --d_model 512 \
  --learning_rate 0.1 \
  --dropout_rate 0.2 \
  --optimizer momo \
  --model baseline \
  --use_attention_mask \
  --data_dir avg_data_env \
  --output_dir results/baseline_avg_data_env_attn_mask