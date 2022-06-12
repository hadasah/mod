SPECS = {
            "transformer_lm_gpt3_small": {
                'NUM_GPUS': 16,
                "NUM_MOD_GPUS": 2,
                "NUM_STEPS": 80000,
                "TRAIN_HOURS": 48,
                "MOD_FROM_STEPS": [24000, 56000, 80000],
                "SAVE_INTERVAL_UPDATES": 8000,
                "LR": 5e-4,
                "UF": 32
            },
            "transformer_lm_gpt3_medium": {
                "NUM_GPUS": 32,
                "NUM_MOD_GPUS": 4,
                "NUM_STEPS": 32000,
                "MOD_FROM_STEPS": [8000, 14000],
                "TRAIN_HOURS": 48,
                "SAVE_INTERVAL_UPDATES": 2000,
                "LR": 5e-4,
                "UF": 32
            },
            "transformer_lm_gpt3_large": {
                "NUM_GPUS": 64,
                "NUM_MOD_GPUS": 8,
                # "NUM_STEPS": 32000,
                # "MOD_FROM_STEPS": [8000, 14000],
                # "TRAIN_HOURS": 48,
                # "SAVE_INTERVAL_UPDATES": 2000,
                # "LR": 5e-4,
                # "UF": 32
            },
            "transformer_lm_gpt3_xl": 128
}
EVAL_FOLDERS = {
        "transformer_lm_gpt3_small": {
            # "dense": "/checkpoint/suching/mod_baselines/MODEL=transformerlmgpt3small_NUMGPUS=16_EXPERIMENT=dense_NUMSTEPS=80000_UPDATEFREQ=32_LR=0.0005/",
            "dense": "/checkpoint/margaretli/mod_publication/NUMGPUS=16_EXPERIMENT=dense_NUMSTEPS=80000_UPDATEFREQ=32_LR=0.0005/",
            "demix": "/checkpoint/suching/mod_baselines/NUMGPUS=16_EXPERIMENT=demix_NUMSTEPS=80000_UPDATEFREQ=32_LR=0.0005/",
            "mod": "/checkpoint/margaretli/mod/_modular_transformer_lm_gpt3_small/modular_transformer_lm_gpt3_small_LR=0.0005{}/"
            # "mod": "/checkpoint/margaretli/mod/suchin_gpt3_small"
        },
        "transformer_lm_gpt3_medium": {
            # "dense": "/checkpoint/suching/mod_baselines/MODEL=transformerlmgpt3medium_NUMGPUS=32_EXPERIMENT=dense_NUMSTEPS=32000_UPDATEFREQ=32_LR=0.0005/",
            "dense": "/checkpoint/margaretli/mod_publication/NUMGPUS=32_EXPERIMENT=dense_NUMSTEPS=32000_UPDATEFREQ=32_LR=0.0005/",
            "demix": "/checkpoint/margaretli/mod_publication/NUMGPUS=32_EXPERIMENT=demix_NUMSTEPS=32000_UPDATEFREQ=32_LR=0.0005/",
            "mod": "/checkpoint/margaretli/mod/_modular_transformer_lm_gpt3_medium/modular_transformer_lm_gpt3_medium_LR=0.0005/"
        }
        
}

INIT_MODEL_FOLDERS = {
    'transformer_lm_gpt3_small': {
        'default': '/checkpoint/suching/fp16/dense_small/',
        "redo": '/checkpoint/margaretli/mod_publication/NUMGPUS=16_EXPERIMENT=dense_NUMSTEPS=80000_UPDATEFREQ=32_LR=0.0005/',
        '0': '/checkpoint/margaretli/mod_publication/NUMGPUS=16_EXPERIMENT=dense_NUMSTEPS=80000_UPDATEFREQ=32_LR=0.0005_DOMAINIDS=0/',
        '1': '/checkpoint/margaretli/mod_publication/NUMGPUS=16_EXPERIMENT=dense_NUMSTEPS=80000_UPDATEFREQ=32_LR=0.0005_DOMAINIDS=1/',
        '2': '/checkpoint/margaretli/mod_publication/NUMGPUS=16_EXPERIMENT=dense_NUMSTEPS=80000_UPDATEFREQ=32_LR=0.0005_DOMAINIDS=2/',
        '3': '/checkpoint/margaretli/mod_publication/NUMGPUS=16_EXPERIMENT=dense_NUMSTEPS=80000_UPDATEFREQ=32_LR=0.0005_DOMAINIDS=3/',
        '4': '/checkpoint/margaretli/mod_publication/NUMGPUS=16_EXPERIMENT=dense_NUMSTEPS=80000_UPDATEFREQ=32_LR=0.0005_DOMAINIDS=4/',
        '5': '/checkpoint/margaretli/mod_publication/NUMGPUS=16_EXPERIMENT=dense_NUMSTEPS=80000_UPDATEFREQ=32_LR=0.0005_DOMAINIDS=5/',
        '6': '/checkpoint/margaretli/mod_publication/NUMGPUS=16_EXPERIMENT=dense_NUMSTEPS=80000_UPDATEFREQ=32_LR=0.0005_DOMAINIDS=6/',
        '7': '/checkpoint/margaretli/mod_publication/NUMGPUS=16_EXPERIMENT=dense_NUMSTEPS=80000_UPDATEFREQ=32_LR=0.0005_DOMAINIDS=7/',
        'unbalanced': '/checkpoint/margaretli/mod_publication/NUMGPUS=16_EXPERIMENT=unbalanced_NUMSTEPS=80000_UPDATEFREQ=32_LR=0.0005/',
    },
    'transformer_lm_gpt3_medium': {
        'default': '/checkpoint/suching/mod_baselines/MODEL=transformerlmgpt3medium_NUMGPUS=32_EXPERIMENT=dense_NUMSTEPS=32000_UPDATEFREQ=32_LR=0.0005/',
        'redo': '/checkpoint/margaretli/mod_publication/NUMGPUS=32_EXPERIMENT=dense_NUMSTEPS=32000_UPDATEFREQ=32_LR=0.0005/',

    }
}

TRAIN_DOMAINS = [
    '1b', 'anonymized_openwebtext', 'anonymized_realnews', 'anonymized_reviews',
    'cs', 'legal', 'med', 'reddit']
VALID_DOMAINS = TRAIN_DOMAINS + ['anonymized_latest_news_redo', 'anonymized_tweets_redo',
    'anonymized_yelp_reviews_redo', 'cord19-redo', 'github_redo', 'gutenberg', 'legal_contracts', 'qasper']
SECONDARY_DOMAINS = ['wikipedia', 'c4', 'realnews', 'code_contests', 'openwebtext', 'Medicine', 'Books', 'JavaScript', 'HTML', 'stackoverflow', 'twitter', 'us_text_20200604']
