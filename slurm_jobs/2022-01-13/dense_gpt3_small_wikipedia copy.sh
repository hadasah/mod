#!/bin/bash
#SBATCH --job-name=dense-gpt3-small-wikipedia
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=zlab
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=40
#SBATCH --mem=320G
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=1

# I use source to initialize conda into the right environment.
source ~/.bashrc

cat $0
echo "--------------------"
cd /gscratch/zlab/margsli/gitfiles/demix

export DATA_DIR=/gscratch/zlab/margsli/data-bin/wikipedia-en/wikipedia-en-0:/gscratch/zlab/margsli/data-bin/wikipedia-en/wikipedia-en-1:/gscratch/zlab/margsli/data-bin/wikipedia-en/wikipedia-en-2:/gscratch/zlab/margsli/data-bin/wikipedia-en/wikipedia-en-3:/gscratch/zlab/margsli/data-bin/wikipedia-en/wikipedia-en-4:/gscratch/zlab/margsli/data-bin/wikipedia-en/wikipedia-en-5:/gscratch/zlab/margsli/data-bin/wikipedia-en/wikipedia-en-6:/gscratch/zlab/margsli/data-bin/wikipedia-en/wikipedia-en-7:/gscratch/zlab/margsli/data-bin/wikipedia-en/wikipedia-en-8:/gscratch/zlab/margsli/data-bin/wikipedia-en/wikipedia-en-9
export NUM_GPUS=8
export PORT=12345
export ARCH=transformer_lm_gpt3_small
export EXPERIMENT=dense
export DATA_PATH=${DATA_DIR}
export FILE_SUFFIX=wikipedia
export SERIALIZATION_DIR=$(pwd)/models/dense_gpt3_small_wikipedia
export NUM_NODES=$((${NUM_GPUS}/8))
# name of wandb project to track model output (at wandb.ai)
WANDB_PROJECT=gpt3_experiments;

TOKENS_PER_SAMPLE=1024;
BATCH_SIZE=2;
LOG_INTERVAL=50;
KEEP_INTERVAL_UPDATES=-1;
LR=5e-4;
CLIP_NORM=0.1;
UPDATE_FREQ=8;
NUM_STEPS=300000;
SAVE_INTERVAL_UPDATES=3000;
VALIDATION_INTERVAL=3000;
NUM_WARMUP_STEPS=$((${NUM_STEPS} * 8 / 100));

fairseq-train $DATA_PATH \
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
          --wandb-project $WANDB_PROJECT  \
          --required-batch-size-multiple 1 \
          --memory-efficient-fp16 \
          --distributed-world-size $NUM_GPUS \
          --distributed-port $PORT \
          --ddp-backend no_c10d \
          --all-gather-list-size 32000;
