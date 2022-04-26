# Number of GPUs you'd like to train on
NUM_GPUS=$1;
# Number of nodes you'd like to train on (assuming 8 GPUs per node) -- adding 7 to round up
NUM_NODES=$(((${NUM_GPUS}+7)/8));
# Fairseq model name (e.g. transformer_lm; see https://github.com/kernelmachine/demix/blob/main/fairseq/models/transformer_lm.py for other options)
ARCH=$2;
# folder data is stored in
TOP_DATA_PATH=$3;
# id of demix domain to train on 
DOMAIN_ID=$4;

PARAMS_TO_FREEZE=$5;

OLD_DIR=$6
# path to top-level directory to where you'd like to output the model
SERIALIZATION_DIR=$7

# Name of subdirectory for this sweep -- should be unique to this sweep
SUBFOLDER_NAME=$8

LOAD_FROM_STEP=$9

SERIALIZATION_DIR=$SERIALIZATION_DIR/LOAD_FROM_STEP=$LOAD_FROM_STEP

NUM_STEPS=${10}

UPDATE_FREQ=${11};
LR=${12};
# name of wandb project to track model output (at wandb.ai)
WANDB_PROJECT=${13};
# name of wandb entity 
WANDB_ENTITY=${14};

MOD_FOLDER=${15};


IDS_TO_DOMAINS=('1b' 'anonymized_openwebtext' 'anonymized_realnews' 'anonymized_reviews' 'cs' 'legal' 'med' 'reddit');
DOMAIN=${IDS_TO_DOMAINS[$DOMAIN_ID]};
DATA_PATH=$TOP_DATA_PATH/$DOMAIN;
# SERIALIZATION_DIR=$MODEL_DIR/DOMAIN_ID=$DOMAIN_ID;

TOKENS_PER_SAMPLE=1024;
BATCH_SIZE=2;
LOG_INTERVAL=50;
KEEP_INTERVAL_UPDATES=-1;

if [[ $ARCH == *"gpt3_small"* ]]; then
     CLIP_NORM=0.1;
     SAVE_INTERVAL_UPDATES=6000;
     VALIDATION_INTERVAL=3000;
     NUM_WARMUP_STEPS=$((${NUM_STEPS} * 8 / 100));
elif [[ $ARCH == *"gpt3_medium"* ]]; then
     NUM_WARMUP_STEPS=$((${NUM_STEPS} * 8 / 100));
     SAVE_INTERVAL_UPDATES=3000;
     VALIDATION_INTERVAL=2000;
     CLIP_NORM=0.1;
elif [[ $ARCH == *"gpt3_large"* ]]; then
     NUM_WARMUP_STEPS=$((${NUM_STEPS} * 8 / 100));
     SAVE_INTERVAL_UPDATES=2000;
     VALIDATION_INTERVAL=1000;
     CLIP_NORM=0.1;
elif [[ $ARCH == *"gpt3_xl"* ]]; then
     NUM_WARMUP_STEPS=$((${NUM_STEPS} * 8 / 100));
     SAVE_INTERVAL_UPDATES=2000;
     VALIDATION_INTERVAL=500;
     CLIP_NORM=0.1;
elif [[ $ARCH == *"transformer_lm"* ]]; then
     TOKENS_PER_SAMPLE=1024;
     CLIP_NORM=0.1;
     SAVE_INTERVAL_UPDATES=12000;
     VALIDATION_INTERVAL=6000;
     NUM_WARMUP_STEPS=$((${NUM_STEPS} * 8 / 100));
fi;

RESET_DATALOADER_PHRASE='--reset-dataloader';
# TRAIN_FROM_SCRATCH_PHRASE='';

# if [[ $OLD_DIR != "None" ]]; then
#      RESET_DATALOADER_PHRASE='--reset-dataloader';
#      python $MOD_FOLDER/mod_utils/mod_checkpoint_utils.py \
#           --old-folder $OLD_DIR \
#           --new-folder $SERIALIZATION_DIR \
#           --subfolder $SUBFOLDER_NAME \
#           --num-steps $LOAD_FROM_STEP \
#           --domain-id $DOMAIN_ID;
# # else
# fi;

python $MOD_FOLDER/fairseq_cli/train.py  $DATA_PATH \
     --arch $ARCH    \
     --task language_modeling \
     --wandb-project $WANDB_PROJECT \
     --save-dir /checkpoint/suching/mod_sweep/_modular_gpt3_small_36K/modular_gpt3_small_36K_LR=0.001/_EXPERIMENT=dense_NUMSTEPS=36000_LR=0.001/DOMAIN_ID=0_NUM_STEPS=18000   \
     --params-to-freeze $PARAMS_TO_FREEZE \
     $RESET_DATALOADER_PHRASE \
     --sample-break-mode none \
     --log-format simple  \
     --log-interval $LOG_INTERVAL    \
     --skip-invalid-size-inputs-valid-test     \
     --validate-interval-updates $VALIDATION_INTERVAL     \
     --save-interval-updates $SAVE_INTERVAL_UPDATES     \
     --keep-interval-updates $KEEP_INTERVAL_UPDATES    \
     --criterion cross_entropy     \
     --lr-scheduler polynomial_decay     \
     --num-workers 2 \
     --max-sentences $BATCH_SIZE \
     --no-epoch-checkpoints \
     --lr $LR              \
     --max-sentences-valid $BATCH_SIZE \
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
     --batch-size-valid 2            \
     --required-batch-size-multiple 1 \
     --memory-efficient-fp16 \
     --ddp-backend no_c10d \
     --distributed-world-size $NUM_GPUS \
     --distributed-port 12345 \
     --all-gather-list-size 32000 ;
