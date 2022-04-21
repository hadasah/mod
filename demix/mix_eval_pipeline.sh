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

# IDS_TO_DOMAINS=('1b' 'anonymized_openwebtext' 'anonymized_realnews' 'anonymized_reviews' 'cs' 'legal' 'med' 'reddit' 'anonymized_latest_news_redo' 'anonymized_tweets_redo' 'anonymized_yelp_reviews_redo' 'cord19-redo' 'github_redo' 'gutenberg' 'legal_contracts' 'qasper');
IDS_TO_DOMAINS=('2007scape' 'History' 'SQL' 'barstoolsports' 'cord19'  'financialindependence'  'insanepeoplefacebook'  'miss_text_20200604'  'pcmasterrace'  'stackoverflow'  'unpopularopinion' 'AMAZON_FASHION' 'Home_and_Kitchen' 'Shell'  'baseball'  'cs' 'fireemblem' 'interestingasfuck' 'mo_text_20200604'  'personalfinance' 'starcitizen' 'us_text_20200604' 'All_Beauty'  'Industrial_and_Scientific'  'Sociology'  'bjj'  'cscareerquestions' 'fla_text_20200604' 'investing' 'mont_text_20200604'  'pics'  'starterpacks' 'utah_text_20200604' 'Appliances'  'Java'  'Software' 'blogsnark' 'cursedcomments' 'fo76'  'iowa_text_20200604'  'motorcycles'  'pokemon'  'stopdrinking' 'va_text_20200604' 'Arts_Crafts_and_Sewing'  'JavaScript'  'Sports_and_Outdoors'  'boardgames' 'cursedimages' 'forhonor' 'ireland' 'movies'  'polisci'  'survivor'  'vancouver' 'Assembly'  'Julia' 'TeX' 'books' 'dankmemes' 'formula1' 'italy'  'nba'  'politics' 'sysadmin'  'vegan' 'Automotive'  'Kindle_Store' 'Tools_and_Home_Improvement'  'bostonceltics'  'dataisbeautiful' 'france'  'jailbreak' 'nd_text_20200604'  'popheads' 'technology'  'vi_text_20200604' 'Biology' 'Lua' 'Toys_and_Games'  'boxoffice' 'dating_advice'  'freefolk' 'kan_text_20200604' 'neb_text_20200604' 'pr_text_20200604' 'techsupport' 'videos' 'Books' 'Luxury_Beauty'  'Video_Games'  'brasil'  'datingoverthirty'  'funny' 'keto' 'neoliberal' 'quityourbullshit' 'teenagers' 'vocab'.'bpe' 'Business'  'Magazine_Subscriptions' 'airsoft'  'btc'  'dauntless' 'ga_text_20200604'  'kpop' 'nev_text_20200604' 'raisedbynarcissists'  'television'  'vt_text_20200604' 'C'  'Markdown'  'ala_text_20200604' 'buildapc'  'dc_text_20200604'  'gameofthrones' 'ky_text_20200604' 'news' 'realnews' 'tenn_text_20200604'  'w-va_text_20200604'  'Mathematics' 'alaska_text_20200604' 'c4' 'de' 'gaming'  'la_text_20200604' 'newzealand' 'reddeadredemption'  'teslamotors' 'wallstreetbets' 'C'++  'Medicine'  'am'-'samoa_text_20200604' 'cal_text_20200604' 'deadbydaylight' 'gardening'  'lakers' 'nfl'  'reddevils'  'tex_text_20200604' 'wash_text_20200604' 'CDs_and_Vinyl' 'Movies_and_TV'  'anime'  'canada'  'del_text_20200604' 'gatekeeping'  'leagueoflegends'  'nh_text_20200604'  'relationship_advice'  'thebachelor' 'weddingplanning' 'CSS'  'Musical_Instruments' 'apexlegends'  'cars' 'depression' 'gifs'  'legaladvice' 'niceguys'  'relationships' 'thedivision' 'whatisthisthing' 'Cell_Phones_and_Accessories'  'Office_Products'  'apple'  'cats'  'golf'  'loseit' 'nj_text_20200604'  'ri_text_20200604' 'therewasanattempt' 'wholesomememes' 'Chemistry' 'PHP' 'argentina'  'changemyview' 'dndnext' 'gonewild' 'manga'  'nottheonion'  'runescape'  'tifu' 'whowouldwin' 'Clothing_Shoes_and_Jewelry' 'Patio_Lawn_and_Garden' 'ariz_text_20200604' 'childfree' 'elderscrollsonline'  'grandorder' 'marvelstudios'  'ny_text_20200604'  'rupaulsdragrace' 'todayilearned'  'wikipedia' 'Digital_Music' 'Perl'  'army' 'churning'  'encoder'.'json' 'gtaonline'  'mass_text_20200604'  'oculus'  'samharris'  'toronto' 'wis_text_20200604' 'Economics' 'Pet_Supplies' 'art' 'classicwow' 'entitledparents' 'guns'  'materials' 'oddlysatisfying' 'sc_text_20200604' 'torontoraptors' 'worldnews' 'Electronics' 'Philosophy'  'askgaybros' 'code_contests'  'env_sci' 'haw_text_20200604' 'md_text_20200604' 'offmychest' 'science'  'totalwar'  'worldpolitics' 'Engineering' 'Physics' 'asoiaf' 'colo_text_20200604'  'europe'  'hearthstone'  'me_irl' 'ohio_text_20200604'  'sd_text_20200604' 'traaaaaaannnnnnnnnns'  'wow' 'FORTRAN' 'PowerShell'  'asoiafcirclejerk'  'comedyheaven' 'exmormon'  'hiphopheads'  'me_text_20200604' 'okla_text_20200604'  'sex' 'trashy' 'wyo_text_20200604' 'GO' 'Prime_Pantry' 'assholedesign' 'confession' 'explainlikeimfive' 'hockey'  'memes'  'openwebtext'  'singapore'  'trees'  'xboxone' 'Geography' 'Psychology'  'atheism'  'confessions'  'facepalm'  'iamatotalpieceofshit' 'mich_text_20200604'  'or_text_20200604'  'smashbros'  'tumblr' 'yugioh' 'Geology' 'Python'  'australia'  'conn_text_20200604'  'fantasybaseball' 'idaho_text_20200604'  'mildlyinfuriating' 'pa_text_20200604'  'soccer' 'twitter' 'HTML' 'Ruby'  'aww' 'conspiracy' 'fatlogic'  'ind_text_20200604' 'mildlyinteresting' 'pathofexile'  'space' 'ukpolitics' 'Haskell' 'Rust'  'bangtan'  'copypasta' 'ffxiv' 'india' 'minn_text_20200604'  'pcgaming' 'stackexchange' 'unitedkingdom');

target_domain=${IDS_TO_DOMAINS[$target_domain_ID]}

model=${ROOT_MODEL_FOLDER}/_EXPERIMENT=dense_NUMSTEPS=48000_LR=0.0005/checkpoint_best.pt;
if [[ "$model_type" == "demix" ]]; then
    for i in $(seq 2 2 15); do 
        if ([[ "$exclude_expert" != "True" ]] || [[ "$i" != "$target_domain_ID" ]]) && ([[ "$only_use_expert" != "True" ]] || [[ "$i" == "$target_domain_ID" ]]) && [[ "${model_checkpoint_ids[$i]}" != "None" ]]; then
            model=${model}:${ROOT_MODEL_FOLDER}/${MODEL_FOLDER}/checkpoint_best-rank-${i}.pt; 
        fi;
    done
elif [[ "$model_type" == "modular" ]]; then
    for i in $(seq 0 2 15); do 
        if ([[ "$exclude_expert" != "True" ]] || [[ "$i" != "$target_domain_ID" ]])  && ([[ "$only_use_expert" != "True" ]] || [[ "$i" == "$target_domain_ID" ]]) && [[ "${model_checkpoint_ids[$i]}" != "None" ]]; then
            model=${model}:${ROOT_MODEL_FOLDER}/${MODEL_FOLDER}${i}/checkpoint_best.pt;
        fi;    
    done
fi;
#model="${model#?}";

evals_folder=generalist_evals_top${eval_top_k};
if [[ "$generalist_model" != "None" ]]; then
    model=${model}:$generalist_model;
    evals_folder=generalist_evals_top${eval_top_k};
fi;
evals_folder=generalist_${evals_folder}_${id};

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
