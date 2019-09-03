#!/bin/bash
# This script is for running our GLUE approach.
# Usage: bash run_glue.sh ${TASK} ${GLUEDATA} ${SEED} ${GPU_ID}
#   - TASK: task name list delimited by ",". Defaults to all.
#   - GLUEDATA: GLUE data directory. Defaults to "data".
#   - LOGPATH: log directory. Defaults to "logs".
#   - SEED: random seed. Defaults to 111.
#   - GPU_ID: GPU to use, or -1 for CPU. Defaults to 0.
#   - MODEL: pretrained model to use. Defaults to 'xlnet-base-cased'


TASK=${1:-CoLA,MNLI,MRPC,QNLI,QQP,RTE,SST-2,STS-B,WNLI}
GLUEDATA=${2:-data}
LOGPATH=${3:-logs}
SEED=${4:-1}
GPU=${5:-0}
MODEL=${5:-xlnet-base-cased}

# 
python run.py \
  --task ${TASK} \
  --seed ${SEED} \
  --data_dir ${GLUEDATA} \
  --log_path ${LOGPATH} \
  --device ${GPU} \
  --bert_model ${MODEL} \
  --n_epochs 3 \
  --train_split train \
  --valid_split dev \
  --optimizer adam \
  --lr 2e-5 \
  --grad_clip 1.0 \
  --warmup_percentage 0.0 \
  --counter_unit epoch \
  --evaluation_freq 0.1 \
  --checkpoint_freq 1 \
  --checkpointing 1 \
  --checkpoint_metric model/train/all/loss:min \
  --checkpoint_task_metrics CoLA/GLUE/dev/matthews_corrcoef:max,MNLI/GLUE/dev/accuracy:max,MRPC/GLUE/dev/accuracy_f1:max,QNLI/GLUE/dev/accuracy:max,QQP/GLUE/dev/accuracy_f1:max,RTE/GLUE/dev/accuracy:max,SNLI/GLUE/dev/accuracy:max,SST-2/GLUE/dev/accuracy:max,STS-B/GLUE/dev/pearson_spearman:max,WNLI/GLUE/dev/accuracy:max \
  --checkpoint_runway 0.5 \
  --checkpoint_clear True \
  --batch_size 16 \
  --max_sequence_length 200