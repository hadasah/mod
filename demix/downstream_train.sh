
SWEEP_NAME=$1;
# Number of GPUs you'd like to train on
NUM_GPUS=$2;
# Number of nodes you'd like to train on (assuming 8 GPUs per node) -- adding 7 to round up
NUM_NODES=$(((${NUM_GPUS}+7)/8));
# Fairseq model name (e.g. transformer_lm; see https://github.com/kernelmachine/demix/blob/main/fairseq/models/transformer_lm.py for other options)
ARCH=$3;
# Baseline type: choice between demix, dense, unbalanced_dense, and domain_token
EXPERIMENT=$4;
# id of demix domain to train on 
DOMAIN_ID=$5;
# path to top level directory to where you'd like to output the model
TOP_LEVEL_SERIALIZATION_DIR=$6;
# name of wandb project to track model output (at wandb.ai)
WANDB_PROJECT=$7;

PARAMS_TO_FREEZE=$8;

INIT_MODEL_TOP_LEVEL_FOLDER=$9;
INIT_MODEL_CKPT_NUM=${10};

CONTINUE_TRAIN=${11};

MODEL_ID=${12};

IDS_TO_DOMAINS=('1b' 'anonymized_openwebtext' 'anonymized_realnews' 'anonymized_reviews' 'cs' 'legal' 'med' 'reddit');

DOMAIN=${IDS_TO_DOMAINS[$DOMAIN_ID]}

if [[ $DOMAIN == *"wikipedia"* ]]; then
    DATA_PATH=/gscratch/zlab/margsli/data-bin/wikipedia-en/wikipedia-en-all/wikipedia-en-all/;
else
    DATA_PATH=/gscratch/zlab/margsli/gitfiles/demix-data/data-bin/${DOMAIN};
fi
SERIALIZATION_DIR=${TOP_LEVEL_SERIALIZATION_DIR}/${SWEEP_NAME}${MODEL_ID};

if [[ $CONTINUE_TRAIN == "True" ]]; then
    INIT_MODEL_PHRASE="";
else
    if [[ $EXPERIMENT == *"demix"* ]]; then
        INIT_MODEL=${INIT_MODEL_TOP_LEVEL_FOLDER}/checkpoint_1_${INIT_MODEL_CKPT_NUM}-rank-${DOMAIN_ID}.pt;
    else
        INIT_MODEL=${INIT_MODEL_TOP_LEVEL_FOLDER}/checkpoint_1_${INIT_MODEL_CKPT_NUM}.pt;
    fi
    INIT_MODEL_PHRASE="--finetune-from-model $INIT_MODEL "
          
fi
TOKENS_PER_SAMPLE=1024;
BATCH_SIZE=2;
LOG_INTERVAL=50;
KEEP_INTERVAL_UPDATES=-1;

LR=1e-5;
CLIP_NORM=0.1;
UPDATE_FREQ=8;
NUM_STEPS=300000;
SAVE_INTERVAL_UPDATES=6000;
VALIDATION_INTERVAL=3000;
NUM_WARMUP_STEPS=$((${NUM_STEPS} * 8 / 100));

fairseq-train    $DATA_PATH \
          $INIT_MODEL_PHRASE \
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
          --save-dir ${SERIALIZATION_DIR}    \
          --batch-size-valid 2                        \
          --wandb-project $WANDB_PROJECT \
          --required-batch-size-multiple 1 \
          --memory-efficient-fp16 \
          --ddp-backend no_c10d \
          --params-to-freeze $PARAMS_TO_FREEZE \
          --all-gather-list-size 32000;
