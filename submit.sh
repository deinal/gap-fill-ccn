#!/bin/bash

# Directories
RESULTS_DIR="baseline_5"
DATA_DIRS=("data/two_week_seq")
SEQUENCE=("two_week_seq")

# Models
MODELS=("baseline")
GAPT_MODES=("default" "naive")

# General parameters
EPOCHS=60
BATCH_SIZE=256
DROPOUT_RATE=0.2
LEARNING_RATE=0.01
OPTIMIZER="momo"

# Start experiments
for idx in "${!DATA_DIRS[@]}"; do
  DATA_DIR=${DATA_DIRS[$idx]}
  SUFFIX=${SEQUENCE[$idx]}
  for MODEL in "${MODELS[@]}"; do
    if [ "$MODEL" == "gapt" ]; then
      for GAPT_MODE in "${GAPT_MODES[@]}"; do
        sbatch batch/gapt.sh -a ${BATCH_SIZE} -b ${EPOCHS} -c 8 -d 6 -e 128 -f 256 -g ${LEARNING_RATE} -h ${DROPOUT_RATE} -i ${OPTIMIZER} -j ${MODEL} -k ${GAPT_MODE} -l ${DATA_DIR} -m ${RESULTS_DIR}/${MODEL}_${GAPT_MODE}_${SUFFIX[$i]}
      done
    elif [ "$MODEL" == "lstm" ] || [ "$MODEL" == "gru" ]; then
      sbatch batch/rnn.sh -a ${BATCH_SIZE} -b ${EPOCHS} -c 3 -d 128 -e ${LEARNING_RATE} -f ${DROPOUT_RATE} -g ${OPTIMIZER} -h ${MODEL} -i ${DATA_DIR} -j ${RESULTS_DIR}/${MODEL}_${SUFFIX[$i]}
    elif [ "$MODEL" == "baseline" ]; then
      sbatch batch/baseline.sh -a ${BATCH_SIZE} -b ${EPOCHS} -c 128 -d 512 -e ${LEARNING_RATE} -f ${DROPOUT_RATE} -g ${OPTIMIZER} -h ${MODEL} -i ${DATA_DIR} -j ${RESULTS_DIR}/${MODEL}_${SUFFIX[$i]}
    elif [ "$MODEL" == "mlp" ]; then
      sbatch batch/mlp.sh -a ${BATCH_SIZE} -b ${EPOCHS} -c ${LEARNING_RATE} -d ${DROPOUT_RATE} -e ${OPTIMIZER} -f ${MODEL} -g ${DATA_DIR} -h ${RESULTS_DIR}/${MODEL}_${SUFFIX[$i]}
    fi
    sleep 0.1
  done
done
