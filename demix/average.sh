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
# path to top-level directory to where you'd like to output the model
SERIALIZATION_DIR=$5
# Name of subdirectory for this sweep -- should be unique to this sweep
SUBFOLDER_NAME=$6
# proportion of time to spend in first phase of training -- determines which checkpoint to load
LOAD_FROM_STEP=$7
AVERAGE=${8};
TOPK=${9};
UNIFORM=${10};
WEIGHTS=${11};
PORT=${12};
MOD_FOLDER=${13};
RUN_ID=${14};


DOMAIN=$DOMAIN_ID;
DATA_PATH=$TOP_DATA_PATH;

AVERAGE_PHRASE='';
if [[ $AVERAGE == "True" ]]; then
     AVERAGE_PHRASE="--average";
fi;

UNIFORM_PHRASE='';
if [[ $UNIFORM == "True" ]]; then
     UNIFORM_PHRASE="--uniform";
fi;

NEW_SUBFOLDER_PHRASE='';
if [[ $RUN_ID != "" ]]; then
     NEW_SUBFOLDER_PHRASE="--new-subfolder $RUN_ID ";
fi;
prior_results_path=${SERIALIZATION_DIR}/average_eval/LOADFROMSTEP=${LOAD_FROM_STEP}_DOMAINID=${DOMAIN_ID}_TOPK=${TOPK}_UNIFORM=${UNIFORM}_AVERAGE=${AVERAGE}/dev_posteriors.jsonl;
results_path=${SERIALIZATION_DIR}/average_eval/LOADFROMSTEP=${LOAD_FROM_STEP}_DOMAINID=${DOMAIN_ID}_TOPK=${TOPK}_UNIFORM=${UNIFORM}_AVERAGE=${AVERAGE}/test_results.txt;

mkdir -p ${SERIALIZATION_DIR}/average_eval/LOADFROMSTEP=${LOAD_FROM_STEP}_DOMAINID=${DOMAIN_ID}_TOPK=${TOPK}_UNIFORM=${UNIFORM}_AVERAGE=${AVERAGE};
cd $MOD_FOLDER;
estimator=;
if [[ $ARCH == "transformer_lm_gpt3_small" ]]; then
          modelid="transformerlmgpt3small";
          ROOT_ESTIMATOR_FOLDER='/checkpoint/margaretli/mod/_modular_transformer_lm_gpt3_small/modular_transformer_lm_gpt3_small_LR=0.0005'
elif [[ $ARCH == "transformer_lm_gpt3_medium" ]]; then
     modelid="transformerlmgpt3medium";
     ROOT_ESTIMATOR_FOLDER='/checkpoint/margaretli/mod/_modular_transformer_lm_gpt3_medium/modular_transformer_lm_gpt3_medium_LR=0.0005';
elif [[ $ARCH == "transformer_lm_gpt3_large" ]]; then
     modelid="transformerlmgpt3large";
     ROOT_ESTIMATOR_FOLDER='';
fi; 


if  [[ $UNIFORM == "False" ]]; then
     for i in $(seq 0 7); do 
          estimator=${estimator}:${ROOT_ESTIMATOR_FOLDER}/MODEL=${modelid}_DOMAINID=${i}_LOADFROMSTEP=${LOAD_FROM_STEP}_RESETITEMS=dataloader_UPDATEFREQ=32_LR=0.0005/checkpoint_last.pt
     done
     estimator="${estimator#?}";

     echo $estimator;

     echo "estimating probabilities...";
     target_eval_split=valid_${DOMAIN_ID};
     python -u fairseq_cli/ensemble_eval_lm.py $DATA_PATH \
     --path $estimator \
     --gen-subset $target_eval_split \
     --target-domain train_${DOMAIN_ID} \
     --target-eval ${target_eval_split} \
     --task multidomain_language_modeling \
     --sample-break-mode none \
     --tokens-per-sample 1024      \
     --batch-size 2  \
     --optimizer adafactor \
     --sample-break-mode none     \
     --log-format simple     \
     --log-interval 50     \
     --skip-invalid-size-inputs-valid-test               \
     --criterion cross_entropy     \
     --lr 5e-4        \
     --weight-decay 0.1     \
     --update-freq 1 \
     --clip-norm 0.0     \
     --no-save           \
     --bucket-cap-mb 200                       \
     --ddp-backend no_c10d      \
     --arch transformer_lm                 \
     --train-domains ${DOMAIN_ID} \
     --eval-domains ${DOMAIN_ID} \
     --log-format tqdm \
     --train-subset train_${DOMAIN_ID} \
     --partial-load \
     --ensemble-type "updating_prior" \
     --results-path ${prior_results_path} \
     --max-samples 100 \
     --distributed-world-size $NUM_GPUS \
     --distributed-port $PORT;
     precomputed_prior=$(tail -n 1 ${prior_results_path} | jq -rc '.exp_avg_posterior | join(",")');
     echo $precomputed_prior
     python $MOD_FOLDER/mod_utils/average.py --model-dir $ROOT_ESTIMATOR_FOLDER --output-dir $SERIALIZATION_DIR/$RUN_ID --weights $precomputed_prior --load-from-step ${LOAD_FROM_STEP} --topk $TOPK ${AVERAGE_PHRASE} ${UNIFORM_PHRASE};
else 
     python $MOD_FOLDER/mod_utils/average.py --output-dir $SERIALIZATION_DIR/$RUN_ID --topk $TOPK $UNIFORM_PHRASE --load-from-step $LOAD_FROM_STEP;
fi;
#    --additional-domains c4 Biology wikipedia gutenberg HTML JavaScript twitter stackexchange Java cord19 stackoverflow C C++ Books Physics Mathematics;