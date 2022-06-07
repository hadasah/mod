# Number of GPUs you'd like to evaluate on. Set this equal to number of experts you'd like to mix.
num_gpus=$1
num_nodes=$2
# Path to data-bins
data_path=$3

ROOT_MODEL_FOLDER=$4

MODEL_FOLDER=$5

CHECKPOINT_IDS=$6

ORIGINAL_EXPERTS=$7
ADDITIONAL_EXPERTS=$8

# target domain to evaluate on
target_domain_ID=$9
# Ensemble type, one of "simple_average","cached_prior", "updating_prior", "uniform_prior"
ensemble_type=${10}

model_type=${11}

generalist_model=${12}
eval_top_k=${13}

id=${14}

num_steps=${15}

exclude_expert=${16}

only_use_expert=${17} 

arch=${18}

MOD_FOLDER=${19}

jq_path=${20}

echo $model_type
OIFS=$IFS;
IFS=','
read -a model_checkpoint_ids <<< "$CHECKPOINT_IDS";
IFS=$OIFS;


OIFS=$IFS;
IFS=','
read -a original_experts <<< "$ORIGINAL_EXPERTS";
IFS=$OIFS;


OIFS=$IFS;
IFS=','
read -a additional_experts <<< "$ADDITIONAL_EXPERTS";
IFS=$OIFS;

# IDS_TO_DOMAINS=('1b' 'anonymized_openwebtext' 'anonymized_realnews' 'anonymized_reviews' 'cs' 'legal' 'med' 'reddit' 'anonymized_latest_news_redo' 'anonymized_tweets_redo' 'anonymized_yelp_reviews_redo' 'cord19-redo' 'github_redo' 'gutenberg' 'legal_contracts' 'qasper');

target_domain=$target_domain_ID

model=;

if [[ "$model_type" == "demix" ]]; then
    for i in $(seq 0 2 15); do
        if ([[ "$exclude_expert" != "True" ]] || [[ "$i" != "$target_domain_ID" ]]) && ([[ "$only_use_expert" != "True" ]] || [[ "$i" == "$target_domain_ID" ]]) && [[ "${model_checkpoint_ids[$i]}" != "None" ]]; then
            model=${model}:${ROOT_MODEL_FOLDER}/${MODEL_FOLDER}/checkpoint_best-rank-${i}.pt; 
        fi;
    done
elif [[ "$model_type" == "modular" ]]; then
    if [[ "$generalist_model" != "None" ]]; then
	start=1;
    else start=0;
    fi
    for i in $(seq 0 7); do 
        if ([[ "$exclude_expert" != "True" ]] || [[ "$i" != "$target_domain_ID" ]])  && ([[ "$only_use_expert" != "True" ]] || [[ "$i" == "$target_domain_ID" ]]) && [[ "${model_checkpoint_ids[$i]}" != "None" ]]; then
            # /checkpoint/suching/suchin_mod/small/_EXPERIMENT\=dense_NUMSTEPS\=36000_LR\=0.001/_DOMAIN_3_MOD_STEPS_30000_PHASE1_DENSE
            #model=${model}:${ROOT_MODEL_FOLDER}/${MODEL_FOLDER}/_DOMAIN_${i}_MOD_STEPS_${num_steps}_PHASE1_DENSE/checkpoint_last.pt
            # model=${model}:${ROOT_MODEL_FOLDER}/${MODEL_FOLDER}/DOMAIN_${i}/${num_steps}/checkpoint_${model_checkpoint_ids[$i]}-rank-${i}.pt
#/checkpoint/suching/suchin_mod//small//_EXPERIMENT=demix_mod_NUMSTEPS=36000_LR=0.001/DOMAIN_3/6000/
            # model=${model}:${ROOT_MODEL_FOLDER}/${MODEL_FOLDER}/DOMAIN_${i}/$num_steps/checkpoint_last-rank-0.pt

            if [[ $arch == "transformer_lm_gpt3_small" ]]; then
                modelid="transformerlmgpt3small";
            elif [[ $arch == "transformer_lm_gpt3_medium" ]]; then
                modelid="transformerlmgpt3medium";
            fi;

            model=${model}:${ROOT_MODEL_FOLDER}/MODEL=${modelid}_DOMAINID=${original_experts[$i]}_LOADFROMSTEP=${num_steps}_RESETITEMS=dataloader_UPDATEFREQ=32_LR=0.0005/checkpoint_${model_checkpoint_ids[$i]}.pt;

            # /checkpoint/suching/mod_publication/mod/small/PHASE1_16GPU_MOD_2GPU_DOMAIN_7_MOD_STEPS_72000_PHASE1_DENSE
            # if [[ $i == 7 ]]; then
                # model=${model}:${ROOT_MODEL_FOLDER}/${MODEL_FOLDER}/MOD_2_GPU_DOMAIN_1_MOD_STEPS_FROM_SCRATCH/checkpoint_${model_checkpoint_ids[$i]}.pt;
            # else 
            # model=${model}:${ROOT_MODEL_FOLDER}/${MODEL_FOLDER}/MOD_2_GPU_DOMAIN_${i}_MOD_STEPS_FROM_SCRATCH/checkpoint_${model_checkpoint_ids[$i]}.pt;
            # fi;
            # model=${model}:${ROOT_MODEL_FOLDER}/${MODEL_FOLDER}/PHASE1_16GPU_MOD_2GPU_DOMAIN_${i}_MOD_STEPS_${num_steps}_PHASE1_DENSE/checkpoint_${model_checkpoint_ids[$i]}.pt;
        fi;    
    done

    for i in $(seq 0 31); do
        if ([[ "$exclude_expert" != "True" ]] || [[ "$i" != "$target_domain_ID" ]])  && ([[ "$only_use_expert" != "True" ]] || [[ "$i" == "$target_domain_ID" ]]) && [[ "${model_checkpoint_ids[$i]}" != "None" ]]; then
            # /checkpoint/suching/suchin_mod/small/_EXPERIMENT\=dense_NUMSTEPS\=36000_LR\=0.001/_DOMAIN_3_MOD_STEPS_30000_PHASE1_DENSE
            #model=${model}:${ROOT_MODEL_FOLDER}/${MODEL_FOLDER}/_DOMAIN_${i}_MOD_STEPS_${num_steps}_PHASE1_DENSE/checkpoint_last.pt
            # model=${model}:${ROOT_MODEL_FOLDER}/${MODEL_FOLDER}/DOMAIN_${i}/${num_steps}/checkpoint_${model_checkpoint_ids[$i]}-rank-${i}.pt
#/checkpoint/suching/suchin_mod//small//_EXPERIMENT=demix_mod_NUMSTEPS=36000_LR=0.001/DOMAIN_3/6000/
            # model=${model}:${ROOT_MODEL_FOLDER}/${MODEL_FOLDER}/DOMAIN_${i}/$num_steps/checkpoint_last-rank-0.pt

            if [[ $arch == "transformer_lm_gpt3_small" ]]; then
                modelid="transformerlmgpt3small";
            elif [[ $arch == "transformer_lm_gpt3_medium" ]]; then
                modelid="transformerlmgpt3medium";
            fi;
            model=${model}:/checkpoint/suching/mod/_adaptation_transformer_lm_gpt3_small/adaptation_transformer_lm_gpt3_small_LR=0.0005/MODEL=${modelid}_DOMAINID=${additional_experts[$i]}_LOADFROMSTEP=-1_RESETITEMS=meters,dataloader,optimizer,lr-scheduler_AVERAGE=True_UPDATEFREQ=32_LR=5e-05/checkpoint_${model_checkpoint_ids[$i]}.pt;
            # model=${model}:${ROOT_MODEL_FOLDER}/MODEL=${modelid}_DOMAINID=${i}_LOADFROMSTEP=${num_steps}_RESETITEMS=dataloader_UPDATEFREQ=32_LR=0.0005/checkpoint_${model_checkpoint_ids[$i]}.pt;

            # /checkpoint/suching/mod_publication/mod/small/PHASE1_16GPU_MOD_2GPU_DOMAIN_7_MOD_STEPS_72000_PHASE1_DENSE
            # if [[ $i == 7 ]]; then
                # model=${model}:${ROOT_MODEL_FOLDER}/${MODEL_FOLDER}/MOD_2_GPU_DOMAIN_1_MOD_STEPS_FROM_SCRATCH/checkpoint_${model_checkpoint_ids[$i]}.pt;
            # else 
            # model=${model}:${ROOT_MODEL_FOLDER}/${MODEL_FOLDER}/MOD_2_GPU_DOMAIN_${i}_MOD_STEPS_FROM_SCRATCH/checkpoint_${model_checkpoint_ids[$i]}.pt;
            # fi;
            # model=${model}:${ROOT_MODEL_FOLDER}/${MODEL_FOLDER}/PHASE1_16GPU_MOD_2GPU_DOMAIN_${i}_MOD_STEPS_${num_steps}_PHASE1_DENSE/checkpoint_${model_checkpoint_ids[$i]}.pt;
        fi;    
    done
fi;
model="${model#?}";

evals_folder=evals_top${eval_top_k};
if [[ "$generalist_model" != "None" ]]; then
    model=${model}:$generalist_model;
    evals_folder=generalist_evals_top${eval_top_k};
fi;
evals_folder=${evals_folder}_${id};

prior_results_path=${ROOT_MODEL_FOLDER}/${evals_folder}/${target_domain}/dev_posteriors.jsonl;
results_path=${ROOT_MODEL_FOLDER}/${evals_folder}/${target_domain}/test_results.txt;

mkdir -p ${ROOT_MODEL_FOLDER}/${evals_folder}/${target_domain};
cd $MOD_FOLDER;
# echo $results_path;
echo $model;
# echo $MOD_FOLDER;

echo "estimating probabilities...";
target_eval_split=valid_${target_domain};
 python -u fairseq_cli/ensemble_eval_lm.py $data_path \
--path $model \
--gen-subset $target_eval_split \
--target-domain train_${target_domain} \
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
--train-domains ${target_domain} \
--eval-domains ${target_domain} \
--log-format tqdm \
--train-subset train_${target_domain} \
--partial-load \
--ensemble-type "updating_prior" \
--results-path ${prior_results_path} \
--max-samples 100 \
--distributed-world-size 32 \
--distributed-port 12345;

# alias jq=~/jq-linux64;
precomputed_prior=$(tail -n 1 ${prior_results_path} | ${jq_path} -rc '.exp_avg_posterior | join(",")');
echo $precomputed_prior;

target_eval_split=test_${target_domain};
 python -u fairseq_cli/ensemble_eval_lm.py $data_path \
--path $model \
--gen-subset $target_eval_split \
--target-domain train_${target_domain} \
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
--train-domains ${target_domain} \
--eval-domains ${target_domain} \
--log-format tqdm \
--train-subset train_${target_domain} \
--partial-load \
--results-path ${results_path} \
--ensemble-type ${ensemble_type} \
--precomputed-prior ${precomputed_prior} \
--eval-topk ${eval_top_k} \
--distributed-world-size 32 \
--distributed-port 12345 ;
