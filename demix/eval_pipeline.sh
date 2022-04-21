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


#IDS_TO_DOMAINS=('1b' 'anonymized_openwebtext' 'anonymized_realnews' 'anonymized_reviews' 'cs' 'legal' 'med' 'reddit' 'anonymized_latest_news_redo' 'anonymized_tweets_redo' 'anonymized_yelp_reviews_redo' 'cord19-redo' 'github_redo' 'gutenberg' 'legal_contracts' 'qasper');

IDS_TO_DOMAINS=('2007scape' 'History' 'SQL' 'barstoolsports' 'cord19'  'financialindependence'  'insanepeoplefacebook'  'miss_text_20200604'  'pcmasterrace'  'stackoverflow'  'unpopularopinion' 'AMAZON_FASHION' 'Home_and_Kitchen' 'Shell'  'baseball'  'cs' 'fireemblem' 'interestingasfuck' 'mo_text_20200604'  'personalfinance' 'starcitizen' 'us_text_20200604' 'All_Beauty'  'Industrial_and_Scientific'  'Sociology'  'bjj'  'cscareerquestions' 'fla_text_20200604' 'investing' 'mont_text_20200604'  'pics'  'starterpacks' 'utah_text_20200604' 'Appliances'  'Java'  'Software' 'blogsnark' 'cursedcomments' 'fo76'  'iowa_text_20200604'  'motorcycles'  'pokemon'  'stopdrinking' 'va_text_20200604' 'Arts_Crafts_and_Sewing'  'JavaScript'  'Sports_and_Outdoors'  'boardgames' 'cursedimages' 'forhonor' 'ireland' 'movies'  'polisci'  'survivor'  'vancouver' 'Assembly'  'Julia' 'TeX' 'books' 'dankmemes' 'formula1' 'italy'  'nba'  'politics' 'sysadmin'  'vegan' 'Automotive'  'Kindle_Store' 'Tools_and_Home_Improvement'  'bostonceltics'  'dataisbeautiful' 'france'  'jailbreak' 'nd_text_20200604'  'popheads' 'technology'  'vi_text_20200604' 'Biology' 'Lua' 'Toys_and_Games'  'boxoffice' 'dating_advice'  'freefolk' 'kan_text_20200604' 'neb_text_20200604' 'pr_text_20200604' 'techsupport' 'videos' 'Books' 'Luxury_Beauty'  'Video_Games'  'brasil'  'datingoverthirty'  'funny' 'keto' 'neoliberal' 'quityourbullshit' 'teenagers' 'vocab'.'bpe' 'Business'  'Magazine_Subscriptions' 'airsoft'  'btc'  'dauntless' 'ga_text_20200604'  'kpop' 'nev_text_20200604' 'raisedbynarcissists'  'television'  'vt_text_20200604' 'C'  'Markdown'  'ala_text_20200604' 'buildapc'  'dc_text_20200604'  'gameofthrones' 'ky_text_20200604' 'news' 'realnews' 'tenn_text_20200604'  'w-va_text_20200604'  'Mathematics' 'alaska_text_20200604' 'c4' 'de' 'gaming'  'la_text_20200604' 'newzealand' 'reddeadredemption'  'teslamotors' 'wallstreetbets' 'C'++  'Medicine'  'am'-'samoa_text_20200604' 'cal_text_20200604' 'deadbydaylight' 'gardening'  'lakers' 'nfl'  'reddevils'  'tex_text_20200604' 'wash_text_20200604' 'CDs_and_Vinyl' 'Movies_and_TV'  'anime'  'canada'  'del_text_20200604' 'gatekeeping'  'leagueoflegends'  'nh_text_20200604'  'relationship_advice'  'thebachelor' 'weddingplanning' 'CSS'  'Musical_Instruments' 'apexlegends'  'cars' 'depression' 'gifs'  'legaladvice' 'niceguys'  'relationships' 'thedivision' 'whatisthisthing' 'Cell_Phones_and_Accessories'  'Office_Products'  'apple'  'cats'  'golf'  'loseit' 'nj_text_20200604'  'ri_text_20200604' 'therewasanattempt' 'wholesomememes' 'Chemistry' 'PHP' 'argentina'  'changemyview' 'dndnext' 'gonewild' 'manga'  'nottheonion'  'runescape'  'tifu' 'whowouldwin' 'Clothing_Shoes_and_Jewelry' 'Patio_Lawn_and_Garden' 'ariz_text_20200604' 'childfree' 'elderscrollsonline'  'grandorder' 'marvelstudios'  'ny_text_20200604'  'rupaulsdragrace' 'todayilearned'  'wikipedia' 'Digital_Music' 'Perl'  'army' 'churning'  'gtaonline'  'mass_text_20200604'  'oculus'  'samharris'  'toronto' 'wis_text_20200604' 'Economics' 'Pet_Supplies' 'art' 'classicwow' 'entitledparents' 'guns'  'materials' 'oddlysatisfying' 'sc_text_20200604' 'torontoraptors' 'worldnews' 'Electronics' 'Philosophy'  'askgaybros' 'code_contests'  'env_sci' 'haw_text_20200604' 'md_text_20200604' 'offmychest' 'science'  'totalwar'  'worldpolitics' 'Engineering' 'Physics' 'asoiaf' 'colo_text_20200604'  'europe'  'hearthstone'  'me_irl' 'ohio_text_20200604'  'sd_text_20200604' 'traaaaaaannnnnnnnnns'  'wow' 'FORTRAN' 'PowerShell'  'asoiafcirclejerk'  'comedyheaven' 'exmormon'  'hiphopheads'  'me_text_20200604' 'okla_text_20200604'  'sex' 'trashy' 'wyo_text_20200604' 'GO' 'Prime_Pantry' 'assholedesign' 'confession' 'explainlikeimfive' 'hockey'  'memes'  'openwebtext'  'singapore'  'trees'  'xboxone' 'Geography' 'Psychology'  'atheism'  'confessions'  'facepalm'  'iamatotalpieceofshit' 'mich_text_20200604'  'or_text_20200604'  'smashbros'  'tumblr' 'yugioh' 'Geology' 'Python'  'australia'  'conn_text_20200604'  'fantasybaseball' 'idaho_text_20200604'  'mildlyinfuriating' 'pa_text_20200604'  'soccer' 'twitter' 'HTML' 'Ruby'  'aww' 'conspiracy' 'fatlogic'  'ind_text_20200604' 'mildlyinteresting' 'pathofexile'  'space' 'ukpolitics' 'Haskell' 'Rust'  'bangtan'  'copypasta' 'ffxiv' 'india' 'minn_text_20200604'  'pcgaming' 'stackexchange' 'unitedkingdom');

target_domain=${IDS_TO_DOMAINS[$target_domain_ID]}

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
    --distributed-world-size 8 \
    --distributed-port 4323 \
    --results-path ${results_path} \
    --partial-load
fi
