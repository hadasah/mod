# Path to data binary
DATA_PATH=$1
# Target domain to DAPT to
DOMAIN_ID=$2
# Path to model
MODEL_DIR=$3
ARCH=$4
LOAD_FROM_STEP=$5
# Baseline type: either "full" (update all parameters during DAPT) or "feedforward" (update only feedforward network during DAPT)
EXPERIMENT=$6
# output of dapt'ed model
SERIALIZATION_DIR=${7}_DOMAIN_${DOMAIN_ID}
# suffix to append to output model path, e.g. "final" or "test"
FILE_SUFFIX=$8
# phase one ratio
TOTAL_STEPS=$9
WANDB_PROJECT=${10}
UPDATE_FREQ=${11}
LR=${12}
NUM_GPUS=${13}
MOD_FOLDER=${14}
# number of GPUs to train with, we default to eight GPUs
# NUM_GPUS=1
# distributed port
PORT=${15}

NUM_STEPS=$((${TOTAL_STEPS} - ${LOAD_FROM_STEP}))



IDS_TO_DOMAINS=('1b' 'anonymized_openwebtext' 'anonymized_realnews' 'anonymized_reviews' 'cs' 'legal' 'med' 'reddit');
DOMAIN=${IDS_TO_DOMAINS[$DOMAIN_ID]};
domains=${DOMAIN};
train_subset=train;
valid_subset=valid_${DOMAIN};
# wandb project name (to track experiment on wandb.ai)
if [[ $MODEL_DIR == *"demix"* ]]; then 
	CHECKPOINT=$MODEL_DIR/checkpoint_1_${LOAD_FROM_STEP}-rank-${DOMAIN_ID}.pt;
	SERIALIZATION_DIR=${SERIALIZATION_DIR}_MOD_STEPS_${NUM_STEPS}_PHASE1_DEMIX;
elif [[ $MODEL_DIR == *"dense"* ]]; then
	CHECKPOINT=$MODEL_DIR/checkpoint_1_${LOAD_FROM_STEP}.pt;
	SERIALIZATION_DIR=${SERIALIZATION_DIR}_MOD_STEPS_${NUM_STEPS}_PHASE1_DENSE;
elif [[ $MODEL_DIR == "None" ]]; then
    CHECKPOINT="None"
    SERIALIZATION_DIR=${SERIALIZATION_DIR}_MOD_STEPS_FROM_SCRATCH;
fi

TOKENS_PER_SAMPLE=1024;
BATCH_SIZE=2;
LOG_INTERVAL=50;
VALIDATION_INTERVAL=500;


CLIP_NORM=0.1;
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
                        --save-interval-updates $TOTAL_STEPS     \
                        --keep-interval-updates $TOTAL-STEPS     \
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
                        --max-sentences $BATCH_SIZE \
                        --max-sentences-valid $BATCH_SIZE \
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
                        --finetune-from-model $CHECKPOINT \
                        --desynchronize \
                        --untie-parameters feedforward \
                        --sync-type manual \
                        --fp16 \
                        --unbalanced \
                        --pad-to-fixed-length \
                        --data-parallel-groups "0,1,2,3,4,5,6,7" \
                        --distributed-world-size $NUM_GPUS \
                        --distributed-port $PORT;
elif [[ $CHECKPOINT == *"dense"* ]]; then
        python $MOD_FOLDER/fairseq_cli/train.py     \
                $DATA_PATH     \
                --task multidomain_language_modeling     \
                --sample-break-mode none     \
                --log-format simple     \
                --log-interval $LOG_INTERVAL    \
                --skip-invalid-size-inputs-valid-test     \
                --validate-interval-updates $VALIDATION_INTERVAL     \
                --save-interval-updates $TOTAL_STEPS     \
                --keep-interval-updates $TOTAL_STEPS    \
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
                --max-sentences $BATCH_SIZE \
                --max-sentences-valid $BATCH_SIZE \
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
                --finetune-from-model $CHECKPOINT \
                --fp16 \
                --unbalanced \
                --pad-to-fixed-length \
                --distributed-world-size $NUM_GPUS \
                --distributed-port $PORT;
elif [[ $CHECKPOINT == "None" ]]; then
        python $MOD_FOLDER/fairseq_cli/train.py     \
                $DATA_PATH     \
                --task multidomain_language_modeling     \
                --sample-break-mode none     \
                --log-format simple     \
                --log-interval $LOG_INTERVAL    \
                --skip-invalid-size-inputs-valid-test     \
                --validate-interval-updates $VALIDATION_INTERVAL     \
                --save-interval-updates $TOTAL_STEPS     \
                --keep-interval-updates $TOTAL_STEPS    \
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
                --max-sentences $BATCH_SIZE \
                --max-sentences-valid $BATCH_SIZE \
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
                --fp16 \
                --unbalanced \
                --pad-to-fixed-length \
                --distributed-world-size $NUM_GPUS \
                --distributed-port $PORT;
				
fi
