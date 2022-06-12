# Number of GPUs you'd like to evaluate on. Set this equal to number of experts you'd like to mix.
num_gpus=$1
# Path to data-bins
data_path=$2

ROOT_MODEL_FOLDER=$3

SUBFOLDERS=$4

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

MOD_FOLDER=${16}

jq_path=${17}

echo $model_type
OIFS=$IFS;
IFS=':'
read -a model_checkpoint_ids <<< "$CHECKPOINT_IDS";
read -a subfolders <<< "$SUBFOLDERS";
IFS=$OIFS;

IDS_TO_DOMAINS=('1b' 'anonymized_openwebtext' 'anonymized_realnews' 'anonymized_reviews' 'cs' 'legal' 'med' 'reddit' 'anonymized_latest_news_redo' 'anonymized_tweets_redo' 'anonymized_yelp_reviews_redo' 'cord19-redo' 'github_redo' 'gutenberg' 'legal_contracts' 'qasper');

target_domain=${IDS_TO_DOMAINS[$target_domain_ID]}

cd $MOD_FOLDER;

REGEX_NAME_STR=$SUBFOLDERS;
OIFS=$IFS;
IFS=' ';
read -r results_folder model < <(python -u mod_utils/mix_eval_utils.py \
--regex-name-str ${REGEX_NAME_STR} \
--model-folder ${ROOT_MODEL_FOLDER} \
--exclude-expert ${exclude_expert} \
--only-use-expert ${only_use_expert} \
--generalist-model ${generalist_model} \
--model-type ${model_type} \
--checkpoint-ids ${CHECKPOINT_IDS} \
--target-domain-id ${target_domain_ID});
IFS=$OIFS;
echo $results_folder;
echo $model;
mkdir -p ${results_folder}
prior_results_path=${results_folder}/dev_posteriors.jsonl;
results_path=${results_folder}/test_results.txt;

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
--distributed-world-size $num_gpus \
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
--sample-break-mode none     \
--log-format simple     \
--log-interval 50     \
--skip-invalid-size-inputs-valid-test               \
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
--distributed-world-size $num_gpus \
--distributed-port 12345 ;

# --criterion cross_entropy     \
# --lr 5e-4        \
# --weight-decay 0.1     \
# --update-freq 1 \
# --clip-norm 0.0     \
# --optimizer adafactor \