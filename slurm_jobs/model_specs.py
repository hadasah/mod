SPECS = {
            "transformer_lm_gpt3_small": {
                'NUM_GPUS': 24,
                "NUM_MOD_GPUS": 2,
                "NUM_STEPS": 54000,
                "TRAIN_HOURS": 39,
                "MOD_FROM_STEPS": [24000, 56000, 80000],
                "SAVE_INTERVAL_UPDATES": 6000,
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
                'NUM_GPUS': 64,
                "NUM_MOD_GPUS": 8,
                "NUM_STEPS": 24000,
                "TRAIN_HOURS": 48,
                "MOD_FROM_STEPS": [2000,10000,16000],
                "SAVE_INTERVAL_UPDATES": 2000,
                "LR": 5e-4,
                "UF": 32
            },
            "transformer_lm_gpt3_xl": {
                'NUM_GPUS': 128,
                "NUM_MOD_GPUS": 16,
                "NUM_STEPS": 16000,
                "TRAIN_HOURS": 48,
                "MOD_FROM_STEPS": [4000,8000,12000],
                "SAVE_INTERVAL_UPDATES": 2000,
                "LR": 5e-4,
                "UF": 32
            },
}
EVAL_FOLDERS = {
        "transformer_lm_gpt3_small": {
            "dense": "/checkpoint/suching/mod_baselines/MODEL=transformerlmgpt3small_NUMGPUS=16_EXPERIMENT=dense_NUMSTEPS=80000_UPDATEFREQ=32_LR=0.0005/",
            "demix": "/checkpoint/suching/mod_baselines/MODEL=transformerlmgpt3small_NUMGPUS=16_EXPERIMENT=demix_NUMSTEPS=80000_UPDATEFREQ=32_LR=0.0005/",
            "mod": "/checkpoint/suching/mod/_modular_gpt3_small_80K/modular_gpt3_small_80K_LR=0.0005/",
            "dense_24_domains": "/checkpoint/suching/mod_publication/NUMGPUS=24_EXPERIMENT=dense_NUMSTEPS=54000_UPDATEFREQ=32_LR=0.0005/"
        },
        "transformer_lm_gpt3_medium": {
            "dense": "/checkpoint/suching/mod_baselines/MODEL=transformerlmgpt3medium_NUMGPUS=32_EXPERIMENT=dense_NUMSTEPS=32000_UPDATEFREQ=32_LR=0.0005/",
            "demix": "/checkpoint/suching/mod_baselines/MODEL=transformerlmgpt3medium_NUMGPUS=32_EXPERIMENT=demix_NUMSTEPS=32000_UPDATEFREQ=32_LR=0.0005/",
            "mod": "/checkpoint/suching/mod/_modular_transformer_lm_gpt3_medium/modular_transformer_lm_gpt3_medium_LR=0.0005/"
        },
        "transformer_lm_gpt3_large": {
            "dense": "",
            "demix": "/checkpoint/suching/mod_baselines/MODEL=transformerlmgpt3large_NUMGPUS=64_EXPERIMENT=demix_NUMSTEPS=24000_UPDATEFREQ=32_LR=0.0005/",
            "mod": "/checkpoint/suching/mod/_modular_transformer_lm_gpt3_large/modular_transformer_lm_gpt3_large_LR=0.0005/"
        },
        "transformer_lm_gpt3_xl": {
            "dense": "",
            "demix": "",
            "mod": "",
            "dense_32_domains": "/checkpoint/suching/mod_publication/NUMGPUS=128_EXPERIMENT=dense_NUMSTEPS=16000_UPDATEFREQ=32_LR=0.0005/",
        }
}
