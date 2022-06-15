# Number of GPUs you'd like to evaluate on. Set this equal to number of experts you'd like to mix.
num_gpus=$1
# Path to data-bins
data_path=$2

ROOT_MODEL_FOLDER=$3

MODEL_FOLDER=$4

CHECKPOINT_IDS=$5

# target domain to evaluate on
target_domain_ID=$6
# Ensemble type, one of "simple_average","cached_prior", "updating_prior", "uniform_prior"
ensemble_type=$7

model_type=$8

generalist_model=$9

eval_top_k=${10}

id=${11}

num_steps=${12}

exclude_expert=${13}

only_use_expert=${14} 

arch=${15}

best_average_output_dir=${16}

MOD_FOLDER=${17}

jq_path=${18}

RANDOM_JOB_PORT=${19}

DEV_POSTERIORS_DIR=${20}

ADDITIONAL_DOMAINS=${21}

OIFS=$IFS;
IFS=','
read -a additional_domains <<< "$ADDITIONAL_DOMAINS";
IFS=$OIFS;



declare -A PORTMAP=( ["1b"]=18029 ["cs"]=13365 ["anonymized_realnews"]=62711 ["anonymized_reviews"]=16279 ["anonymized_openwebtext"]=18628 ["med"]=13493 ["reddit"]=51606 ["legal"]=37723 ["cord19-redo"]=27446 ["qasper"]=56775 ["legal_contracts"]=9404 ["github_redo"]=22625 ["gutenberg"]=26581 ["anonymized_latest_news_redo"]=16161 ["anonymized_tweets_redo"]=5261 ["anonymized_yelp_reviews_redo"]=60978 ["Business"]=39568 ["c4"]=27365 ["Sociology"]=25980 ["polisci"]=56025 ["Geology"]=56700 ["materials"]=34434 ["Engineering"]=2435 ["Economics"]=52405 ["Chemistry"]=20252 ["Psychology"]=19438 ["Geography"]=11353 ["C++"]=59095 ["C"]=44430 ["Java"]=5014 ["Mathematics"]=33650 ["Physics"]=42991 ["env_sci"]=45978 ["HTML"]=10421 ["JavaScript"]=19516 ["Biology"]=4345 ["twitter"]=9785 ["stackoverflow"]=4615 ["wikipedia"]=7072 )

declare -A PORTMAPSTEP=( [1000]=100 [16000]=200 [40000]=300 [56000]=400 )

PORT=$(( PORTMAP[$target_domain_ID] + PORTMAPSTEP[${num_steps}] + $RANDOM_JOB_PORT ))
echo $PORT

echo $model_type
OIFS=$IFS;
IFS=','
read -a model_checkpoint_ids <<< "$CHECKPOINT_IDS";
IFS=$OIFS;

# IDS_TO_DOMAINS=('1b' 'anonymized_openwebtext' 'anonymized_realnews' 'anonymized_reviews' 'cs' 'legal' 'med' 'reddit' 'anonymized_latest_news_redo' 'anonymized_tweets_redo' 'anonymized_yelp_reviews_redo' 'cord19-redo' 'github_redo' 'gutenberg' 'legal_contracts' 'qasper');

target_domain=$target_domain_ID

model=;

if [[ "$model_type" == "demix" ]]; then
    for i in $(seq 0 8 63); do
        if ([[ "$exclude_expert" != "True" ]] || [[ "$i" != "$target_domain_ID" ]]) && ([[ "$only_use_expert" != "True" ]] || [[ "$i" == "$target_domain_ID" ]]) && [[ "${model_checkpoint_ids[$i]}" != "None" ]]; then
            model=${model}:${ROOT_MODEL_FOLDER}/${MODEL_FOLDER}/checkpoint_best-rank-${i}.pt; 
        fi;
    done
elif [[ "$model_type" == "modular" ]]; then
    if [[ "$generalist_model" != "None" ]]; then
	start=1;
    else start=0;
    fi
    for i in $(seq $start 7); do 
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
            elif [[ $arch == "transformer_lm_gpt3_large" ]]; then
		        modelid="transformerlmgpt3large";
            fi;

            model=${model}:${ROOT_MODEL_FOLDER}/MODEL=${modelid}_DOMAINID=${i}_LOADFROMSTEP=${num_steps}_RESETITEMS=dataloader_UPDATEFREQ=32_LR=0.0005/checkpoint_${model_checkpoint_ids[$i]}.pt;

            # /checkpoint/suching/mod_publication/mod/small/PHASE1_16GPU_MOD_2GPU_DOMAIN_7_MOD_STEPS_72000_PHASE1_DENSE
            # if [[ $i == 7 ]]; then
                # model=${model}:${ROOT_MODEL_FOLDER}/${MODEL_FOLDER}/MOD_2_GPU_DOMAIN_1_MOD_STEPS_FROM_SCRATCH/checkpoint_${model_checkpoint_ids[$i]}.pt;
            # else 
            # model=${model}:${ROOT_MODEL_FOLDER}/${MODEL_FOLDER}/MOD_2_GPU_DOMAIN_${i}_MOD_STEPS_FROM_SCRATCH/checkpoint_${model_checkpoint_ids[$i]}.pt;
            # fi;
            # model=${model}:${ROOT_MODEL_FOLDER}/${MODEL_FOLDER}/PHASE1_16GPU_MOD_2GPU_DOMAIN_${i}_MOD_STEPS_${num_steps}_PHASE1_DENSE/checkpoint_${model_checkpoint_ids[$i]}.pt;
        fi;    
    done

    for i in $(seq 0 23); do
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
            #MODEL\=transformerlmgpt3small_DOMAINID\=Sociology_LOADFROMSTEP\=-1_RESETITEMS\=optimizer\,meters\,lr-scheduler\,dataloader_NUMSTEPS\=40000_AVERAGE\=False_UPDATEFREQ\=32_LR\=5e-05
            model=${model}:/checkpoint/suching/mod/_adaptation_${arch}/adaptation_${arch}_LR=0.0005/MODEL=${modelid}_DOMAINID=${additional_domains[$i]}_LOADFROMSTEP=-1_RESETITEMS=optimizer,meters,lr-scheduler,dataloader_NUMSTEPS=40000_AVERAGE=False_UPDATEFREQ=32_LR=5e-05/checkpoint_${model_checkpoint_ids[$i]}.pt;
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
evals_folder=${evals_folder}_${id}_LOAD_FROM_STEP_${num_steps};

prior_results_path=${DEV_POSTERIORS_DIR}/${target_domain}/dev_posteriors.jsonl;
best_average_output_dir=${best_average_output_dir}_LOAD_FROM_STEP_${num_steps}
results_path=${best_average_output_dir}/${evals_folder}/${target_domain}/test_results.txt;

mkdir -p ${best_average_output_dir}/${evals_folder}/${target_domain};
cd $MOD_FOLDER;
# echo $results_path;
echo $model;
# echo $MOD_FOLDER;


# alias jq=~/jq-linux64;
precomputed_prior=$(tail -n 1 ${prior_results_path} | ${jq_path} -rc '.exp_avg_posterior | join(",")');
echo $precomputed_prior;

target_eval_split=test_${target_domain};
 python -u fairseq_cli/eval_lm_average.py $data_path \
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
--best-average-output-dir ${best_average_output_dir}/DOMAINID=${target_domain};
