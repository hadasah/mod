
# Number of GPUs you'd like to train on
NUM_GPUS=$1
# Number of nodes you'd like to train on (assuming 8 GPUs per node)
NUM_NODES=$((${NUM_GPUS}/8))
# Distributed port
# Fairseq model name (e.g. transformer_lm; see https://github.com/kernelmachine/demix/blob/main/fairseq/models/transformer_lm.py for other options)
ARCH=$2
# Baseline type: choice between demix, dense, unbalanced_dense, and domain_token
EXPERIMENT=$3
# Path to data-bins
DATA_PATH=$4

PARAMS_TO_FREEZE=$5;
# Old directory to copy checkpoints from -- can be "None" if training from scratch
OLD_DIR=$6
# path to top-level directory to where you'd like to output the model
SERIALIZATION_DIR=$7
# Name of subdirectory for this sweep -- should be unique to this sweep
SUBFOLDER_NAME=$8
# Must be either "None" or comma-separated list of some subset of [meters, dataloader, optimizer, lr-scheduler]
PHASE_ONE_RATIO=$9

RESET_ITEMS=${10}
SERIALIZATION_DIR=$SERIALIZATION_DIR/$SUBFOLDER_NAME

NUM_STEPS=${11};
UPDATE_FREQ=${12};
LR=${13};
# name of wandb project to track model output (at wandb.ai)
WANDB_PROJECT=${14};
# name of wandb entity 
WANDB_ENTITY=${15};

MOD_FOLDER=${16};

DOMAIN_IDS=${17};

RUN_ID=${18}

WANDB_PROJECT=gpt3_experiments;

if [[ $DOMAIN_ID == *"all"* ]]; then
     # list of domains you'd like to train on, that can be found in $DATA_PATH
     domains=1b,anonymized_openwebtext,anonymized_realnews,anonymized_reviews,cs,legal,med,reddit;
     # validation datasets for each domain
     valid_subset=valid_1b,valid_cs,valid_legal,valid_med,valid_anonymized_openwebtext,valid_anonymized_realnews,valid_reddit,valid_anonymized_reviews;

     DATA_PHRASE="$DATA_PATH \
          --task multidomain_language_modeling 
          --valid-subset $valid_subset \
          --train-domains $domains  \
          --eval-domains $domains \
          --criterion desynchronized_cross_entropy     \
          "
# name of wandb project to track model output (at wandb.ai)
else
     IDS_TO_DOMAINS=('1b' 'anonymized_openwebtext' 'anonymized_realnews' 'anonymized_reviews' 'cs' 'legal' 'med' 'reddit');
     DATA_PHRASE="";
     OIFS=$IFS;
     IFS=','
     read -a domain_ids <<< "$DOMAIN_IDS";
     IFS=$OIFS;
     for id in "${domain_ids[@]}"; do
          DATA_PHRASE="${DATA_PHRASE}:$DATA_PATH/${IDS_TO_DOMAINS[$id]}"
     done;
     DATA_PHRASE="${DATA_PHRASE#?} \
          --task language_modeling \
          --criterion cross_entropy     \
          ";
fi;
echo $DATA_PHRASE;


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

# $DATA_PARALLEL_GROUPS identifies which ranks we will synchronize over. "A,B C,D" means we will synchronize ranks A,B and synchronize ranks C,D.
if [[ $NUM_GPUS == "8" ]]; then
     if [[ $EXPERIMENT == *"dense"*  || $EXPERIMENT == *"domain_token"* ]]; then
          DATA_PARALLEL_GROUPS="0,1,2,3,4,5,6,7";
     elif  [[ $EXPERIMENT == *"demix"* ]]; then
          DATA_PARALLEL_GROUPS="0 1 2 3 4 5 6 7";
     fi;
elif [[ $NUM_GPUS == "16" ]]; then
     if [[ $EXPERIMENT == *"dense"*  || $EXPERIMENT == *"domain_token"* ]]; then
          DATA_PARALLEL_GROUPS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15";
     elif  [[ $EXPERIMENT == *"demix"* ]]; then
          DATA_PARALLEL_GROUPS="0,1 2,3 4,5 6,7 8,9 10,11 12,13 14,15";
     fi;
elif [[ $NUM_GPUS == "32" ]]; then
     if [[ $EXPERIMENT == *"dense"*  || $EXPERIMENT == *"domain_token"* ]]; then
          DATA_PARALLEL_GROUPS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31";
     elif  [[ $EXPERIMENT == *"demix"* ]]; then
          DATA_PARALLEL_GROUPS="0,1,2,3 4,5,6,7 8,9,10,11 12,13,14,15 16,17,18,19 20,21,22,23 24,25,26,27 28,29,30,31";
     fi;
elif [[ $NUM_GPUS == "64" ]]; then
     if [[ $EXPERIMENT == *"dense"*  || $EXPERIMENT == *"domain_token"* ]]; then
     	DATA_PARALLEL_GROUPS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63";
     elif  [[ $EXPERIMENT == *"demix"* ]]; then
	     DATA_PARALLEL_GROUPS="0,1,2,3,4,5,6,7 8,9,10,11,12,13,14,15 16,17,18,19,20,21,22,23 24,25,26,27,28,29,30,31 32,33,34,35,36,37,38,39 40,41,42,43,44,45,46,47 48,49,50,51,52,53,54,55 56,57,58,59,60,61,62,63";
     fi;
elif [[ $NUM_GPUS == "128" ]]; then
     if [[ $EXPERIMENT == *"dense"*  || $EXPERIMENT == *"domain_token"* ]]; then
     	DATA_PARALLEL_GROUPS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127";
     elif  [[ $EXPERIMENT == *"demix"* ]]; then
	     DATA_PARALLEL_GROUPS="0,1,2,3,4,5,6,7 8,9,10,11,12,13,14,15 16,17,18,19,20,21,22,23 24,25,26,27,28,29,30,31 32,33,34,35,36,37,38,39 40,41,42,43,44,45,46,47 48,49,50,51,52,53,54,55 56,57,58,59,60,61,62,63 64,65,66,67,68,69,70,71 72,73,74,75,76,77,78,79 80,81,82,83,84,85,86,87 88,89,90,91,92,93,94,95 96,97,98,99,100,101,102,103 104,105,106,107,108,109,110,111 112,113,114,115,116,117,118,119 120,121,122,123,124,125,126,127";
     fi;
fi;


RESET_PHRASE='';
DISTRIBUTED_ARGS_PHRASE='';
OIFS=$IFS;
IFS=','
read -a reset_vals <<< "$RESET_ITEMS";
IFS=$OIFS;

if [ $NUM_GPUS \> 1 ]; then
     DISTRIBUTED_ARGS_PHRASE="--ddp-backend no_c10d --distributed-world-size $NUM_GPUS --distributed-port 12345";
fi;

if [[ $OLD_DIR != "None" ]]; then
     NEW_SUBFOLDER_PHRASE='';
     if [[ $RUN_ID != "" ]]; then
          NEW_SUBFOLDER_PHRASE="--new-subfolder $ID ";
     fi;
     python $MOD_FOLDER/mod_utils/mod_checkpoint_utils.py \
          --old-folder $OLD_DIR \
          --new-folder $SERIALIZATION_DIR \
          --subfolder $SUBFOLDER_NAME \
          $NEW_SUBFOLDER_PHRASE \
          --phase-one-ratio $PHASE_ONE_RATIO \
          --domain-id $DOMAIN_ID;
fi;

if [[ $RESET_ITEMS != "None" ]]; then
     for item in "${reset_vals[@]}"; do
          RESET_PHRASE="${RESET_PHRASE}--reset-${item} "
     done;
fi;
echo $RESET_PHRASE;

if [[ $EXPERIMENT == *"demix"* ]]; then
     python $MOD_FOLDER/fairseq_cli/train.py \
          $DATA_PHRASE
          --sample-break-mode none \
          --log-format simple  \
          --log-interval $LOG_INTERVAL    \
          --skip-invalid-size-inputs-valid-test     \
          --validate-interval-updates $VALIDATION_INTERVAL     \
          --save-interval-updates $SAVE_INTERVAL_UPDATES     \
          --keep-interval-updates $KEEP_INTERVAL_UPDATES    \
          --arch $ARCH    \
	     --criterion desynchronized_cross_entropy     \
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
          --save-dir $SERIALIZATION_DIR/$RUN_ID/   \
          --batch-size-valid 2                        \
          --wandb-project $WANDB_PROJECT           \
          --wandb-entity $WANDB_ENTITY \
          --required-batch-size-multiple 1 \
          --fp16 \
          --desynchronize --domain-parallel \
          $DISTRIBUTED_ARGS_PHRASE \
          --sync-type manual \
          --untie-parameters feedforward \
          --data-parallel-groups "${DATA_PARALLEL_GROUPS}" \
          --all-gather-list-size 32000 \
          $RESET_PHRASE \
          --pad-to-fixed-length;
elif [[ $EXPERIMENT == *"unbalanced"* ]]; then
     python $MOD_FOLDER/fairseq_cli/train.py     $DATA_PATH \
          --task multidomain_language_modeling \
          --sample-break-mode none \
          --log-format simple  \
          --log-interval $LOG_INTERVAL    \
          --skip-invalid-size-inputs-valid-test     \
          --validate-interval-updates $VALIDATION_INTERVAL     \
          --save-interval-updates $SAVE_INTERVAL_UPDATES     \
          --keep-interval-updates $KEEP_INTERVAL_UPDATES    \
          --arch $ARCH    \
          --criterion desynchronized_cross_entropy     \
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
          --save-dir $SERIALIZATION_DIR/$RUN_ID/     \
          --batch-size-valid 2                        \
          --wandb-project $WANDB_PROJECT           \
          --valid-subset $valid_subset \
          --train-domains $domains  \
          --eval-domains $domains \
          --required-batch-size-multiple 1 \
          --memory-efficient-fp16 \
          $DISTRIBUTED_ARGS_PHRASE \
          --all-gather-list-size 32000 \
          $RESET_PHRASE \
          --unbalanced;
elif [[ $EXPERIMENT == *"dense"* ]]; then
     python $MOD_FOLDER/fairseq_cli/train.py $DATA_PHRASE \
          --sample-break-mode none \
          --log-format simple  \
          --log-interval $LOG_INTERVAL    \
          --skip-invalid-size-inputs-valid-test     \
          --validate-interval-updates $VALIDATION_INTERVAL     \
          --save-interval-updates $SAVE_INTERVAL_UPDATES     \
          --keep-interval-updates $KEEP_INTERVAL_UPDATES    \
          --arch $ARCH    \
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
          --save-dir $SERIALIZATION_DIR/$RUN_ID/     \
          --batch-size-valid 2                        \
          --wandb-project $WANDB_PROJECT           \
          --required-batch-size-multiple 1 \
          --fp16 \
          $DISTRIBUTED_ARGS_PHRASE \
          $RESET_PHRASE \
          --all-gather-list-size 32000;
elif [[ $EXPERIMENT == *"switch"* ]]; then
     python $MOD_FOLDER/fairseq_cli/train.py $DATA_PATH     \
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
          --criterion moe_cross_entropy     \
          --lr-scheduler polynomial_decay     \
          --lr $LR              \
          --tokens-per-sample 1024          \
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
          --save-dir $SERIALIZATION_DIR/$RUN_ID/         \
          --batch-size-valid 2                        \
          --train-domains $domains \
          --eval-domains $domains \
          --valid-subset $valid_subset \
          --required-batch-size-multiple 1 \
          --update-freq $UPDATE_FREQ \
          --fp16 \
          --fp16-no-flatten-grads \
          --moe-freq 2 \
          --moe-top1-expert \
          --moe-expert-count $NUM_GPUS \
          --moe-gating-use-fp32 \
          --moe-gate-loss-wt 0.01 \
          --moe-gate-loss-combine-method sum \
          --moe-second-expert-policy all \
          $DISTRIBUTED_ARGS_PHRASE \
          $RESET_PHRASE \
          --all-gather-list-size 32000;
elif [[ $EXPERIMENT == *"gshard"* ]]; then
     python $MOD_FOLDER/fairseq_cli/train.py $DATA_PHRASE     \
          --sample-break-mode none     \
          --log-format simple     \
          --log-interval $LOG_INTERVAL    \
          --skip-invalid-size-inputs-valid-test     \
          --validate-interval-updates $VALIDATION_INTERVAL     \
          --save-interval-updates $SAVE_INTERVAL_UPDATES     \
          --keep-interval-updates $KEEP_INTERVAL_UPDATES     \
          --no-epoch-checkpoints \
          --arch $ARCH    \
          --criterion moe_cross_entropy     \
          --lr-scheduler polynomial_decay     \
          --lr $LR               \
          --tokens-per-sample 1024          \
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
          --save-dir $SERIALIZATION_DIR/$RUN_ID/         \
          --batch-size-valid 2                        \
          --train-domains $domains \
          --eval-domains $domains \
          --valid-subset $valid_subset \
          --required-batch-size-multiple 1 \
          --update-freq $UPDATE_FREQ \
          --fp16 \
          --fp16-no-flatten-grads \
          --moe-freq 2 \
          --moe-expert-count $NUM_GPUS \
          --moe-gating-use-fp32 \
          --moe-gate-loss-wt 0.01 \
          --moe-gate-loss-combine-method sum \
          --moe-second-expert-policy all \
          $DISTRIBUTED_ARGS_PHRASE \
          $RESET_PHRASE \
          --all-gather-list-size 32000;
elif [[ $EXPERIMENT == *"domain_token"* ]]; then
     # domain token
     python $MOD_FOLDER/fairseq_cli/train.py     $DATA_PATH \
          --task multidomain_language_modeling \
          --sample-break-mode none \
          --log-format simple  \
          --log-interval $LOG_INTERVAL    \
          --skip-invalid-size-inputs-valid-test     \
          --validate-interval-updates $VALIDATION_INTERVAL     \
          --save-interval-updates $SAVE_INTERVAL_UPDATES     \
          --keep-interval-updates $KEEP_INTERVAL_UPDATES    \
          --arch $ARCH    \
	     --criterion desynchronized_cross_entropy     \
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
          --save-dir $SERIALIZATION_DIR/$RUN_ID/     \
          --batch-size-valid 2                        \
          --wandb-project $WANDB_PROJECT           \
          --valid-subset $valid_subset \
          --train-domains $domains  \
          --eval-domains $domains \
          --required-batch-size-multiple 1 \
          --memory-efficient-fp16 \
          $DISTRIBUTED_ARGS_PHRASE \
          --all-gather-list-size 32000 \
          $RESET_PHRASE \
          --add-domain-token;
fi;
