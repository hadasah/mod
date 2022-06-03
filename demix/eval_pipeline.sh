data_bin=$1

ROOT_MODEL_FOLDER=$2

model_folder=$3

CHECKPOINT_ID=$4
# split you'd like to evaluate on ("valid" or "test")
split=$5
# target domain to evaluate on
target_domain_ID=$6

DEMIX_FOLDER=$7

force_domain_token=$8


# IDS_TO_DOMAINS=('1b' 'anonymized_openwebtext' 'anonymized_realnews' 'anonymized_reviews' 'cs' 'legal' 'med' 'reddit' 'anonymized_latest_news_redo' 'anonymized_tweets_redo' 'anonymized_yelp_reviews_redo' 'cord19-redo' 'github_redo' 'gutenberg' 'legal_contracts' 'qasper');

target_domain=$target_domain_ID

model=${ROOT_MODEL_FOLDER}/${model_folder}/checkpoint_${CHECKPOINT_ID}.pt;

evals_folder=evals

results_path=${ROOT_MODEL_FOLDER}/${model_folder}/${evals_folder}/${target_domain}/test_results.txt

mkdir -p ${ROOT_MODEL_FOLDER}/${model_folder}/${evals_folder}/${target_domain};
cd $DEMIX_FOLDER;

if [[ "$model" == *"domain_token"* ]]; then
    if [[ -z "$force_domain_token" ]]; then
        python -u fairseq_cli/eval_lm.py \
                ${data_bin} \
                --path ${model} \
                --gen-subset ${split}_${target_domain} \
                --task multidomain_language_modeling \
                --sample-break-mode none \
                --tokens-per-sample 1024     \
                --batch-size 2  \
                --original-domains 1b,anonymized_openwebtext,anonymized_realnews,anonymized_reviews,cs,legal,med,reddit \
                --eval-domains ${target_domain} \
                --results-path ${results_path} \
                --partial-load \
                --add-domain-token;
    else
        python -u fairseq_cli/eval_lm.py \
                ${data_bin} \
                --path ${model} \
                --gen-subset ${split}_${target_domain} \
                --task multidomain_language_modeling \
                --sample-break-mode none \
                --tokens-per-sample 1024     \
                --batch-size 2  \
                --eval-domains ${target_domain} \
                --results-path ${results_path} \
                --partial-load \
                --add-domain-token \
                --force-domain-token $force_domain_token;
    fi;

elif [[ "$model" == *"gshard"* || "$model" == *"switch"* ]]; then
    python -u fairseq_cli/eval_lm.py \
        ${data_bin} \
        --path ${model} \
        --gen-subset ${split}_${target_domain} \
        --task multidomain_language_modeling \
        --sample-break-mode none \
        --tokens-per-sample 1024     \
        --batch-size 2  \
        --eval-domains ${target_domain} \
        --results-path ${results_path} \
        --distributed-world-size 64 \
        --distributed-port 4234 \
	--is-moe;
else
    python -u fairseq_cli/eval_lm.py \
    ${data_bin} \
    --path ${model} \
    --gen-subset ${split}_${target_domain} \
    --task multidomain_language_modeling \
    --sample-break-mode none \
    --tokens-per-sample 1024     \
    --batch-size 2  \
    --eval-domains ${target_domain} \
    --results-path ${results_path} \
	--partial-load
fi