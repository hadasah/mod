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

exclude_expert=${12}

only_use_expert=${13} 

MOD_FOLDER=${14}

jq_path=${15}

OIFS=$IFS;
IFS=','
read -a model_checkpoint_ids <<< "$CHECKPOINT_IDS";
IFS=$OIFS;

IDS_TO_DOMAINS=('1b' 'anonymized_openwebtext' 'anonymized_realnews' 'anonymized_reviews' 'cs' 'legal' 'med' 'reddit' 'anonymized_latest_news_redo' 'anonymized_tweets_redo' 'anonymized_yelp_reviews_redo' 'cord19-redo' 'github_redo' 'gutenberg' 'legal_contracts' 'qasper');

target_domain=${IDS_TO_DOMAINS[$target_domain_ID]}

model=;
if [[ "$model_type" == "demix" ]]; then
    for i in $(seq 0 7); do 
        if ([[ "$exclude_expert" != "True" ]] || [[ "$i" != "$target_domain_ID" ]]) && ([[ "$only_use_expert" != "True" ]] || [[ "$i" == "$target_domain_ID" ]]) && [[ "${model_checkpoint_ids[$i]}" != "None" ]]; then
            model=${model}:${ROOT_MODEL_FOLDER}/${MODEL_FOLDER}/checkpoint_${model_checkpoint_ids[$i]}-rank-${i}.pt; 
        fi;
    done
elif [[ "$model_type" == "modular" ]]; then
    for i in $(seq 0 7); do 
        if ([[ "$exclude_expert" != "True" ]] || [[ "$i" != "$target_domain_ID" ]])  && ([[ "$only_use_expert" != "True" ]] || [[ "$i" == "$target_domain_ID" ]]) && [[ "${model_checkpoint_ids[$i]}" != "None" ]]; then
            model=${model}:${ROOT_MODEL_FOLDER}/${MODEL_FOLDER}${i}/checkpoint_${model_checkpoint_ids[$i]}.pt;
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

prior_results_path=${ROOT_MODEL_FOLDER}/${MODEL_FOLDER}/${evals_folder}/${target_domain}/dev_posteriors.jsonl;
results_path=${ROOT_MODEL_FOLDER}/${MODEL_FOLDER}/${evals_folder}/${target_domain}/test_results.txt;

mkdir -p ${ROOT_MODEL_FOLDER}/${MODEL_FOLDER}/${evals_folder}/${target_domain};
cd $MOD_FOLDER;
# echo $results_path;
# echo $model;
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
--distributed-world-size $num_gpus \
--distributed-port 12345 \
--max-samples 100;

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
--distributed-world-size $num_gpus \
--distributed-port 12345 \
--precomputed-prior ${precomputed_prior} \
--eval-topk ${eval_top_k};