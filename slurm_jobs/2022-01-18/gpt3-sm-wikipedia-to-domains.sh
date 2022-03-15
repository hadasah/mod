#!/bin/bash
#SBATCH --job-name=dense-gpt3-small-wikipedia-to-cs-freeze-attn
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=zlab
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=1

# I use source to initialize conda into the right environment.
source ~/.bashrc

cat $0
echo "--------------------"
cd /gscratch/zlab/margsli/gitfiles/demix

# 3000, 18000, 36000, 
export DOMAIN=cs
# export CKPT_UPDATES=
export DATA_DIR=/gscratch/zlab/margsli/gitfiles/demix-data/data-bin/${DOMAIN}
export NUM_GPUS=1
export PORT=12347
export ARCH=transformer_lm_gpt3_small
export EXPERIMENT=dense
export DATA_PATH=${DATA_DIR}
export PARAMS_TO_FREEZE=self_attn
export FILE_SUFFIX=wikipedia_to_${DOMAIN}_freeze_${PARAMS_TO_FREEZE}
export SERIALIZATION_DIR=$(pwd)/models/dense_gpt3_small_wikipedia_to_${DOMAIN}_freeze_${PARAMS_TO_FREEZE}
export NUM_NODES=1
export INIT_MODEL=/gscratch/zlab/margsli/gitfiles/demix/models/dense_gpt3_small_wikipedia/dense_8_GPUs_transformer_lm_gpt3_small_wikipedia/checkpoint_best.pt

WANDB_PROJECT=gpt3_experiments;

TOKENS_PER_SAMPLE=1024;
BATCH_SIZE=2;
LOG_INTERVAL=50;
KEEP_INTERVAL_UPDATES=-1;
LR=5e-5;
CLIP_NORM=0.1;
UPDATE_FREQ=8;
NUM_STEPS=300000;
SAVE_INTERVAL_UPDATES=3000;
VALIDATION_INTERVAL=3000;
NUM_WARMUP_STEPS=$((${NUM_STEPS} * 8 / 100));

python fairseq_cli/train.py     $DATA_PATH \
          --finetune-from-model $INIT_MODEL \
          --task language_modeling \
          --sample-break-mode none \
          --log-format simple  \
          --log-interval $LOG_INTERVAL    \
          --skip-invalid-size-inputs-valid-test     \
          --validate-interval-updates $VALIDATION_INTERVAL     \
          --save-interval-updates $SAVE_INTERVAL_UPDATES     \
          --keep-interval-updates $KEEP_INTERVAL_UPDATES    \
          --arch $ARCH    \
          --criterion cross_entropy     \
          --lr-scheduler polynomial_decay     \
          --num-workers 2 \
          --max-sentences $BATCH_SIZE \
          --no-epoch-checkpoints \
          --max-sentences-valid $BATCH_SIZE \
          --lr $LR              \
          --tokens-per-sample $TOKENS_PER_SAMPLE          \
          --optimizer adam \
          --adam-betas '(0.9, 0.95)'  \
          --adam-eps 10e-8 \
          --weight-decay 0.1 \
          --clip-norm $CLIP_NORM      \
          --max-update $NUM_STEPS     \
          --total-num-update $NUM_STEPS     \
          --warmup-updates $NUM_WARMUP_STEPS     \
          --update-freq $UPDATE_FREQ     \
          --save-dir ${SERIALIZATION_DIR}/dense_${NUM_GPUS}_GPUs_${ARCH}_${FILE_SUFFIX}      \
          --batch-size-valid 2                        \
          --wandb-project $WANDB_PROJECT \
          --required-batch-size-multiple 1 \
          --memory-efficient-fp16 \
          --distributed-world-size $NUM_GPUS \
          --distributed-port $PORT \
          --ddp-backend no_c10d \
          --params-to-freeze $PARAMS_TO_FREEZE \
          --all-gather-list-size 32000;
