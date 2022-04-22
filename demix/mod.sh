# Path to data binary
DATA_PATH=$1
# Target domain to DAPT to
DOMAIN_ID=$2
# Path to model
MODEL_DIR=$3
LOAD_FROM_STEP=$4
# Baseline type: either "full" (update all parameters during DAPT) or "feedforward" (update only feedforward network during DAPT)
EXPERIMENT=$5
# output of dapt'ed model
SERIALIZATION_DIR=$6/DOMAIN_${DOMAIN_ID}
# suffix to append to output model path, e.g. "final" or "test"
FILE_SUFFIX=$7
# phase one ratio
NUM_STEPS=$((36000 - ${LOAD_FROM_STEP}))
WANDB_PROJECT=$8
MOD_FOLDER=$9
# number of GPUs to train with, we default to eight GPUs
NUM_GPUS=8
# distributed port
PORT=12345

SERIALIZATION_DIR=$SERIALIZATION_DIR/$NUM_STEPS
IDS_TO_DOMAINS=('1b' 'anonymized_openwebtext' 'anonymized_realnews' 'anonymized_reviews' 'cs' 'legal' 'med' 'reddit');
DOMAIN=${IDS_TO_DOMAINS[$DOMAIN_ID]};
domains=${DOMAIN};
train_subset=train;
valid_subset=valid_${DOMAIN};
# wandb project name (to track experiment on wandb.ai)
CHECKPOINT=$MODEL_DIR/checkpoint_1_${LOAD_FROM_STEP}-rank-${DOMAIN_ID}.pt
if [[ $MODEL_DIR == *"small"* ]]; then
    ARCH=transformer_lm_gpt3_small;
    LR=1e-4;
elif [[ $MODEL_DIR == *"med"* ]]; then
    ARCH=transformer_lm_gpt3_medium;
    if [[ $ == *"demix"* ]]; then
        NUM_STEPS=1000;
        SAVE_INTERVAL_UPDATES=200;
    elif [[ $ == *"dense"* ]]; then
        NUM_STEPS=750;
        SAVE_INTERVAL_UPDATES=200;
    fi
elif [[ $ == *"gpt3_large"* ]]; then
    ARCH=transformer_lm_gpt3_large;
    if [[ $ == *"demix"* ]]; then
        NUM_STEPS=500;
        SAVE_INTERVAL_UPDATES=200;
    elif [[ $ == *"dense"* ]]; then
        NUM_STEPS=300;
        SAVE_INTERVAL_UPDATES=200;
    fi;
elif [[ $ == *"gpt3_xl"* ]]; then
    ARCH=transformer_lm_gpt3_xl;
    if [[ $ == *"demix"* ]]; then
        NUM_STEPS=1250;
        SAVE_INTERVAL_UPDATES=250;
    elif [[ $ == *"dense"* ]]; then
        NUM_STEPS=1000;
        SAVE_INTERVAL_UPDATES=200;
    fi;
elif [[ $ == *"transformer_lm"* ]]; then
    ARCH=transformer_lm;
    if [[ $ == *"demix"* ]]; then
        NUM_STEPS=1000;
        SAVE_INTERVAL_UPDATES=200;
    elif [[ $ == *"dense"* ]]; then
        NUM_STEPS=750;
        SAVE_INTERVAL_UPDATES=200;
    fi
fi;

TOKENS_PER_SAMPLE=1024;
BATCH_SIZE=2;
LOG_INTERVAL=50;
VALIDATION_INTERVAL=500;


CLIP_NORM=0.1;
UPDATE_FREQ=4;
NUM_WARMUP_STEPS=$((${NUM_STEPS} * 8 / 100));


if [[ $CHECKPOINT == *"demix"* ]]; then
    python $MOD_FOLDER/fairseq_cli/train.py     \
                        $DATA_PATH     \
                        --task multidomain_language_modeling     \
                        --sample-break-mode none     \
                        --log-format simple     \
                        --log-interval $LOG_INTERVAL    \
                        --skip-invalid-size-inputs-valid-test     \
                        --validate-interval-updates $VALIDATION_INTERVAL     \
                        --save-interval-updates 36000     \
                        --keep-interval-updates 36000     \
                        --no-epoch-checkpoints \
                        --arch $ARCH    \
                        --criterion cross_entropy    \
                        --lr-scheduler polynomial_decay     \
                        --lr $LR             \
                        --tokens-per-sample $TOKENS_PER_SAMPLE          \
                        --optimizer adam \
                        --adam-betas '(0.9, 0.95)'  \
                        --adam-eps 10e-8 \
                        --weight-decay 0.1 \
                        --num-workers 2 \
                        --max-sentences 2 \
                        --max-sentences-valid 2 \
                        --clip-norm $CLIP_NORM      \
                        --max-update $NUM_STEPS     \
                        --total-num-update $NUM_STEPS     \
                        --warmup-updates $NUM_WARMUP_STEPS     \
                        --wandb-project $WANDB_PROJECT \
                        --save-dir ${SERIALIZATION_DIR}        \
                        --train-subset $train_subset \
                        --valid-subset $valid_subset \
                        --train-domains $domains  \
                        --eval-domains $domains \
                        --required-batch-size-multiple 1 \
                        --update-freq $UPDATE_FREQ \
                        --dropout 0.0 \
                        --finetune-from-model $CHECKPOINT \
                        --desynchronize \
                        --untie-parameters feedforward \
			--sync-type manual \
			--memory-efficient-fp16 \
    			--unbalanced \
                        --data-parallel-groups "0,1,2,3,4,5,6,7" \
			--distributed-world-size $NUM_GPUS \
			--distributed-port $PORT
                        #--sync-type manual \
                        #--memory-efficient-fp16 \
			            #--unbalanced;
elif [[ $CHECKPOINT == *"dense"* ]]; then
        python fairseq_cli/train.py     \
                $DATA_PATH     \
                --task multidomain_language_modeling     \
                --sample-break-mode none     \
                --log-format simple     \
                --log-interval $LOG_INTERVAL    \
                --skip-invalid-size-inputs-valid-test     \
                --validate-interval-updates $VALIDATION_INTERVAL     \
                --save-interval-updates $SAVE_INTERVAL_UPDATES     \
                --keep-interval-updates $KEEP_INTERVAL_UPDATES     \
                --no-epoch-checkpoints \
                --arch $ARCH    \
                --criterion cross_entropy    \
                --lr-scheduler polynomial_decay     \
                --lr $LR             \
                --tokens-per-sample $TOKENS_PER_SAMPLE          \
                --optimizer adam \
                --adam-betas '(0.9, 0.95)'  \
                --adam-eps 10e-8 \
                --weight-decay 0.1 \
                --num-workers 2 \
                --max-sentences 2 \
                --max-sentences-valid 2 \
                --clip-norm $CLIP_NORM      \
                --max-update $NUM_STEPS     \
                --total-num-update $NUM_STEPS     \
                --warmup-updates $NUM_WARMUP_STEPS     \
                --wandb-project $WANDB_PROJECT \
                --save-dir ${SERIALIZATION_DIR}        \
                --train-subset $train_subset \
                --valid-subset $valid_subset \
                --train-domains $domains  \
                --eval-domains $domains \
                --required-batch-size-multiple 1 \
                --update-freq $UPDATE_FREQ \
                --dropout 0.0 \
                --finetune-from-model $CHECKPOINT \
                --distributed-world-size 8 \
                --memory-efficient-fp16 \
                --distributed-port $PORT \
                --unbalanced;
fi
