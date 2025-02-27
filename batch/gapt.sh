#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=512M
#SBATCH --gres=gpu:v100:1
#SBATCH --output=slurm/%j.out

module load pytorch/1.13

while getopts a:b:c:d:e:f:g:h:i:j:k:l:m: flag
do
    case "${flag}" in
        a) BATCH_SIZE=${OPTARG};;
        b) EPOCHS=${OPTARG};;
        c) N_HEAD=${OPTARG};;
        d) N_LAYERS=${OPTARG};;
        e) D_MODEL=${OPTARG};;
        f) D_FEEDFORWARD=${OPTARG};;
        g) LEARNING_RATE=${OPTARG};;
        h) DROPOUT_RATE=${OPTARG};;
        i) OPTIMIZER=${OPTARG};;
        j) MODEL=${OPTARG};;
        k) MODE=${OPTARG};;
        l) DATA_DIR=${OPTARG};;
        m) OUTPUT_DIR=${OPTARG};;
    esac
done

python train.py \
  --devices 1 \
  --num_workers 32 \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --n_head $N_HEAD \
  --n_layers $N_LAYERS \
  --d_model $D_MODEL \
  --d_feedforward $D_FEEDFORWARD \
  --learning_rate $LEARNING_RATE \
  --dropout_rate $DROPOUT_RATE \
  --optimizer $OPTIMIZER \
  --model $MODEL \
  --mode $MODE \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR
