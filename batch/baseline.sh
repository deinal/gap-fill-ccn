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

while getopts a:b:c:d:e:f:g:h:i:j: flag
do
    case "${flag}" in
        a) BATCH_SIZE=${OPTARG};;
        b) EPOCHS=${OPTARG};;
        c) D_EMBEDDING=${OPTARG};;
        d) D_MODEL=${OPTARG};;
        e) LEARNING_RATE=${OPTARG};;
        f) DROPOUT_RATE=${OPTARG};;
        g) OPTIMIZER=${OPTARG};;
        h) MODEL=${OPTARG};;
        i) DATA_DIR=${OPTARG};;
        j) OUTPUT_DIR=${OPTARG};;
    esac
done

python train.py \
  --devices 1 \
  --num_workers 32 \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --d_embedding $D_EMBEDDING \
  --d_model $D_MODEL \
  --learning_rate $LEARNING_RATE \
  --dropout_rate $DROPOUT_RATE \
  --optimizer $OPTIMIZER \
  --model $MODEL \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR
