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
# proportion of time to spend in first phase of training -- determines which checkpoint to load
LOAD_FROM_STEP=$9
# Must be either "None" or comma-separated list of some subset of [meters, dataloader, optimizer, lr-scheduler]
RESET_ITEMS=${10}
# SERIALIZATION_DIR=$SERIALIZATION_DIR/$PHASE_ONE_RATIO

NUM_STEPS=${11};
AVERAGE=${12};
AVERAGE_WEIGHTS=${13};
STOP_TIME_HOURS=${14};
UPDATE_FREQ=${15};
LR=${16};

RANDOM_JOB_PORT=${17};

declare -A PORTMAP=( ["Sociology"]=16329 ["Business"]=19003 ["polisci"]=50839 ["Geology"]=49897 ["materials"]=18320 ["Engineering"]=54242 ["Economics"]=21730 ["Chemistry"]=37977 ["Psychology"]=1352 ["Geography"]=24592 ["C++"]=17794 ["C"]=63176 ["Java"]=38823 ["Mathematics"]=7751 ["Physics"]=31368 ["env_sci"]=15844 ["HTML"]=49077 ["JavaScript"]=8862 ["Biology"]=7772 ["twitter"]=3697 ["stackoverflow"]=8409 ["wikipedia"]=53889 ["c4"]=64589 ["gutenberg"]=24598 ["1b_demix_paper"]=32488 ["cs_demix_paper"]=40595 ["anonymized_realnews_demix_paper"]=10957 ["anonymized_reviews_demix_paper"]=36811 ["anonymized_openwebtext_demix_paper"]=58278 ["med_demix_paper"]=42877 ["reddit_demix_paper"]=14469 ["legal_demix_paper"]=11202 ["supreme_court"]=6763 ["dm_mathematics"]=27530 ["opensubtitles"]=55418 ["uspto"]=7326 ["2021_newscrawl"]=15295 ["na"]=50844 ["stories"]=33818 ["hackernews"]=22028 ["bookcorpus"]=36739 ["acl"]=43029 ["Ruby"]=32478 ["Sports_and_Outdoors"]=9295 ["Python"]=11225 ["cord19"]=52883 ["stackexchange"]=63653 ["PHP"]=43443 ["C#"]=7851 ["CSS"]=39792 ["code_contests"]=38677 ["Markdown"]=25798 ["anonymized_yelp_reviews_redo_demix_paper"]=34655 ["GO"]=4359 ["Philosophy"]=31930 ["History"]=38585 ["art"]=52554 ["Movies_and_TV"]=15751 ["Electronics"]=22088 ["Books"]=4013 ["Home_and_Kitchen"]=11661 ["Clothing_Shoes_and_Jewelry"]=56574 ["gaming_comments"]=25052 ["sports_comments"]=6504 )

declare -A PORTMAPSTEP=( [1000]=100 [20000]=50 [16000]=200 [40000]=300 [56000]=400 [8000]=104 )

PORT=$(( PORTMAP[$DOMAIN_ID] + PORTMAPSTEP[${LOAD_FROM_STEP}] + $RANDOM_JOB_PORT ))
echo $PORT

# name of wandb project to track model output (at wandb.ai)
WANDB_PROJECT=${18};
# name of wandb entity 
WANDB_ENTITY=${19};

MOD_FOLDER=${20};

RUN_ID=${21};


#IDS_TO_DOMAINS=('1b' 'anonymized_openwebtext' 'anonymized_realnews' 'anonymized_reviews' 'cs' 'legal' 'med' 'reddit' 'anonymized_latest_news_redo' 'anonymized_tweets_redo' 'anonymized_yelp_reviews_redo' 'cord19-redo' 'github_redo' 'gutenberg' 'legal_contracts' 'qasper');
#DOMAIN=${IDS_TO_DOMAINS[$DOMAIN_ID]};
DOMAIN=$DOMAIN_ID;
DATA_PATH=$TOP_DATA_PATH;

domains=${DOMAIN};
train_subset=train;
valid_subset=valid_${DOMAIN};



TOKENS_PER_SAMPLE=1024;
BATCH_SIZE=2;
LOG_INTERVAL=50;
KEEP_INTERVAL_UPDATES=50;

if [[ $ARCH == *"gpt3_small"* ]]; then
     CLIP_NORM=0.1;
     SAVE_INTERVAL_UPDATES=2000;
     VALIDATION_INTERVAL=500;
elif [[ $ARCH == *"gpt3_medium"* ]]; then
     SAVE_INTERVAL_UPDATES=1000;
     VALIDATION_INTERVAL=250;
     CLIP_NORM=0.1;
elif [[ $ARCH == *"gpt3_large"* ]]; then
     SAVE_INTERVAL_UPDATES=32000;
     VALIDATION_INTERVAL=500;
     CLIP_NORM=0.1;
elif [[ $ARCH == *"gpt3_xl"* ]]; then
     SAVE_INTERVAL_UPDATES=2000;
     VALIDATION_INTERVAL=500;
     CLIP_NORM=0.1;
elif [[ $ARCH == *"transformer_lm"* ]]; then
     TOKENS_PER_SAMPLE=1024;
     CLIP_NORM=0.1;
     SAVE_INTERVAL_UPDATES=12000;
     VALIDATION_INTERVAL=6000;
fi;

RESET_PHRASE='';
DISTRIBUTED_ARGS_PHRASE='';
OIFS=$IFS;
IFS=','
read -a reset_vals <<< "$RESET_ITEMS";
IFS=$OIFS;

if [ $NUM_GPUS \> 1 ]; then
      DISTRIBUTED_ARGS_PHRASE="--ddp-backend no_c10d --distributed-world-size $NUM_GPUS --distributed-port $PORT";
fi;
if [[ $OLD_DIR != "None" ]]; then
      NEW_SUBFOLDER_PHRASE='';
      if [[ $RUN_ID != "" ]]; then
           NEW_SUBFOLDER_PHRASE="--new-subfolder $RUN_ID ";
      fi;
fi;
#      if [[ $AVERAGE == "True" ]]; then

#      prior_results_path=${SERIALIZATION_DIR}/average_eval/${DOMAIN_ID}/dev_posteriors.jsonl;
#      results_path=${SERIALIZATION_DIR}/average_eval/${DOMAIN_ID}/test_results.txt;

#      mkdir -p ${SERIALIZATION_DIR}/average_eval/${DOMAIN_ID};
#      cd $MOD_FOLDER;
#      # echo $results_path;
#      estimator=;
#      ROOT_ESTIMATOR_FOLDER='/checkpoint/suching/mod/_modular_gpt3_small_80K/modular_gpt3_small_80K_LR=0.0005/'
#      for i in $(seq 0 7); do 
#           estimator=${estimator}:${ROOT_ESTIMATOR_FOLDER}/MODEL=transformerlmgpt3small_DOMAINID=${i}_LOADFROMSTEP=24000_RESETITEMS=dataloader_UPDATEFREQ=32_LR=0.0005/checkpoint_last.pt
#     done
#     estimator="${estimator#?}";

#      echo $estimator;

#      echo "estimating probabilities...";
#      target_eval_split=valid_${DOMAIN_ID};
#      python -u fairseq_cli/ensemble_eval_lm.py $DATA_PATH \
#      --path $estimator \
#      --gen-subset $target_eval_split \
#      --target-domain train_${DOMAIN_ID} \
#      --target-eval ${target_eval_split} \
#      --task multidomain_language_modeling \
#      --sample-break-mode none \
#      --tokens-per-sample 1024      \
#      --batch-size 2  \
#      --optimizer adafactor \
#      --sample-break-mode none     \
#      --log-format simple     \
#      --log-interval 50     \
#      --skip-invalid-size-inputs-valid-test               \
#      --criterion cross_entropy     \
#      --lr 5e-4        \
#      --weight-decay 0.1     \
#      --update-freq 1 \
#      --clip-norm 0.0     \
#      --no-save           \
#      --bucket-cap-mb 200                       \
#      --ddp-backend no_c10d      \
#      --arch transformer_lm                 \
#      --train-domains ${DOMAIN_ID} \
#      --eval-domains ${DOMAIN_ID} \
#      --log-format tqdm \
#      --train-subset train_${DOMAIN_ID} \
#      --partial-load \
#      --ensemble-type "updating_prior" \
#      --results-path ${prior_results_path} \
#      --max-samples 100;
#      --distributed-world-size 8;
#      --distributed-port 12345;
     # alias jq=~/jq-linux64;
#      prior_results_path=/checkpoint/suching/mod/_modular_gpt3_small_80K/modular_gpt3_small_80K_LR\=0.0005/evals_top8_Base_dense_LOAD_FROM_STEP_24000_LR_0.0005/${DOMAIN_ID}/dev_posteriors.jsonl;
#      precomputed_prior=$(tail -n 1 ${prior_results_path} | jq -rc '.exp_avg_posterior | join(",")');
#      python $MOD_FOLDER/mod_utils/average.py --output-dir $SERIALIZATION_DIR/$RUN_ID --weights $precomputed_prior
#      #    --additional-domains c4 Biology wikipedia gutenberg HTML JavaScript twitter stackexchange Java cord19 stackoverflow C C++ Books Physics Mathematics;
#     else
     
#     fi;
# fi;


python $MOD_FOLDER/mod_utils/mod_checkpoint_utils.py \
          --old-folder $OLD_DIR \
          --new-folder $SERIALIZATION_DIR \
          --subfolder "DOMAINID=${DOMAIN_ID}" \
          $NEW_SUBFOLDER_PHRASE \
          --load-from-step -1 \
          --domain-id $DOMAIN_ID;

          
if [[ $RESET_ITEMS != "None" ]]; then
     for item in "${reset_vals[@]}"; do
          RESET_PHRASE="${RESET_PHRASE}--reset-${item} "
     done;
fi;
echo $RESET_PHRASE;


python $MOD_FOLDER/fairseq_cli/train.py  $DATA_PATH \
     --arch $ARCH    \
     --task multidomain_language_modeling \
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
     --train-subset $train_subset \
     --valid-subset $valid_subset \
     --train-domains $domains  \
     --eval-domains $domains \
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
     --stop-time-hours $STOP_TIME_HOURS \
     --update-freq $UPDATE_FREQ     \
     --batch-size-valid 2            \
     $DISTRIBUTED_ARGS_PHRASE \
     --required-batch-size-multiple 1 \
     --fp16 \
     --unbalanced \
     --no-epoch-checkpoints \
     --all-gather-list-size 32000;
