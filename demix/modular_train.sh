SWEEP_NAME=$1
# Number of GPUs you'd like to train on
NUM_GPUS=$2
# Number of nodes you'd like to train on (assuming 8 GPUs per node)
NUM_NODES=$((${NUM_GPUS}/8))
# Distributed port
# Fairseq model name (e.g. transformer_lm; see https://github.com/kernelmachine/demix/blob/main/fairseq/models/transformer_lm.py for other options)
ARCH=$3
# Baseline type: choice between demix, dense, unbalanced_dense, and domain_token
EXPERIMENT=$4
# Path to data-bins
DATA_PATH=$5
# Comma separated list of demix domains to train on. "all" or, e.g. "0,1"
DOMAIN_ID=$6;
# Comma separated list of parameters to freeze, or "None"
PARAMS_TO_FREEZE=$7;
# Old directory to copy checkpoints from -- can be "None" if training from scratch
OLD_DIR=$8
# path to top-level directory to where you'd like to output the model
SERIALIZATION_DIR=$9
# Name of subdirectory containing checkpoint to copy
SUBFOLDER_NAME=${10}
# Ratio of updates to spend in first phase training - "None" or a float, e.g. 0.5
PHASE_ONE_RATIO=${11}
# Number of updates to spend in first phase training - "None" or an int, e.g. 36000
PHASE_ONE_UPDATE_NUM=${12}
# comma separated list of items to reset in checkpoint (dataloader,meters,lr-scheduler,optimizer), or "None"
RESET_ITEMS=${13};
# total number of steps in training -- determines lr schedule
NUM_STEPS=${14};
# update frequency
UPDATE_FREQ=${15};
# learning rate
LR=${16};
# name of wandb project to track model output (at wandb.ai)
WANDB_PROJECT=${17};
# name of wandb entity 
WANDB_ENTITY=${18};
# path to mod code folder
MOD_FOLDER=${19};
# Unique identifer of this run
RUN_ID=${20}

# Set data path
IDS_TO_DOMAINS=('1b' 'anonymized_openwebtext' 'anonymized_realnews' 'anonymized_reviews' 'cs' 'legal' 'med' 'reddit');
DOMAIN=${IDS_TO_DOMAINS[$DOMAIN_ID]};
DATA_PATH=$DATA_PATH/$DOMAIN;

# Set training hyperparams
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

# Resetting dataloader,meters,optimizer,lr-scheduler as needed
RESET_PHRASE='';
if [[ $RESET_ITEMS != "None" ]]; then
     OIFS=$IFS;
     IFS=','
     read -a reset_vals <<< "$RESET_ITEMS";
     IFS=$OIFS;
     for item in "${reset_vals[@]}"; do
          RESET_PHRASE="${RESET_PHRASE}--reset-${item} "
     done;
fi;
echo $RESET_PHRASE;

# Add distributed training args if necessary
DISTRIBUTED_ARGS_PHRASE='';
if [ $NUM_GPUS \> 1 ]; then
     DISTRIBUTED_ARGS_PHRASE="--ddp-backend no_c10d --distributed-world-size $NUM_GPUS --distributed-port 12345";
fi;

# Copying over the checkpoint
if [[ $OLD_DIR != "None" ]]; then
     NEW_SUBFOLDER_PHRASE='';
     if [[ $RUN_ID != "" ]]; then
          NEW_SUBFOLDER_PHRASE="--new-subfolder $RUN_ID ";
     fi;
     PHASE_PHRASE='';
     if [[ $PHASE_ONE_RATIO != "None" ]] && [[ $PHASE_ONE_UPDATE_NUM != "None" ]]; then
          printf '%s\n' "Cannot set both PHASE_ONE_RATIO and PHASE_ONE_UPDATE_NUM" >&2
          exit 1
     elif [[ $PHASE_ONE_RATIO != "None" ]]; then
          PHASE_PHRASE="--phase-one-ratio $PHASE_ONE_RATIO"
     elif [[ $PHASE_ONE_UPDATE_NUM != "None" ]]; then 
          PHASE_PHRASE="--phase-one-update-num $PHASE_ONE_UPDATE_NUM"
     else
          printf '%s\n' "If copying checkpoints, must set one of PHASE_ONE_RATIO or PHASE_ONE_UPDATE_NUM" >&2
          exit 1
     fi;
     python $MOD_FOLDER/mod_utils/mod_checkpoint_utils.py \
          --old-folder $OLD_DIR \
          --new-folder $SERIALIZATION_DIR \
          --subfolder $SUBFOLDER_NAME \
          $NEW_SUBFOLDER_PHRASE \
          $PHASE_PHRASE \
          --domain-id $DOMAIN_ID;
fi;

# Train
python $MOD_FOLDER/fairseq_cli/train.py  $DATA_PATH \
     --arch $ARCH    \
     --task language_modeling \
     --wandb-project $WANDB_PROJECT \
     --save-dir $SERIALIZATION_DIR/$RUN_ID/   \
     --params-to-freeze $PARAMS_TO_FREEZE \
     $RESET_PHRASE \
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
     $DISTRIBUTED_ARGS_PHRASE \
     --required-batch-size-multiple 1 \
     --memory-efficient-fp16 \
     --all-gather-list-size 32000 ;
