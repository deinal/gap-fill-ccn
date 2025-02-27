#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=512M
#SBATCH --gres=gpu:v100:1
#SBATCH --output=slurm/%j.out

module load pytorch/1.13

while getopts a:b:c:d:e:f:g:h:i: flag
do
    case "${flag}" in
        a) BATCH_SIZE=${OPTARG};;
        b) EPOCHS=${OPTARG};;
        c) LEARNING_RATE=${OPTARG};;
        d) DROPOUT_RATE=${OPTARG};;
        e) OPTIMIZER=${OPTARG};;
        f) MODEL=${OPTARG};;
        g) DATA_DIR=${OPTARG};;
        h) OUTPUT_DIR=${OPTARG};;
    esac
done

python train.py \
  --devices 1 \
  --num_workers 32 \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --learning_rate $LEARNING_RATE \
  --dropout_rate $DROPOUT_RATE \
  --optimizer $OPTIMIZER \
  --model $MODEL \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR
