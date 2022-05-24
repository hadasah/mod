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
            "transformer_lm_gpt3_large": 64,
            "transformer_lm_gpt3_xl": 128
}
EVAL_FOLDERS = {
        "transformer_lm_gpt3_small": {
            "dense": "/checkpoint/suching/mod_baselines/MODEL=transformerlmgpt3small_NUMGPUS=16_EXPERIMENT=dense_NUMSTEPS=80000_UPDATEFREQ=32_LR=0.0005/",
            "demix": "/checkpoint/suching/mod_baselines/MODEL=transformerlmgpt3small_NUMGPUS=16_EXPERIMENT=demix_NUMSTEPS=80000_UPDATEFREQ=32_LR=0.0005/",
            "mod": "/checkpoint/suching/mod/_modular_gpt3_small_80K/modular_gpt3_small_80K_LR=0.0005/"
        },
        "transformer_lm_gpt3_medium": {
            "dense": "/checkpoint/suching/mod_baselines/MODEL=transformerlmgpt3medium_NUMGPUS=32_EXPERIMENT=dense_NUMSTEPS=32000_UPDATEFREQ=32_LR=0.0005/",
            "demix": "/checkpoint/suching/mod_baselines/MODEL=transformerlmgpt3medium_NUMGPUS=32_EXPERIMENT=demix_NUMSTEPS=32000_UPDATEFREQ=32_LR=0.0005/",
            "mod": "/checkpoint/suching/mod/_modular_transformer_lm_gpt3_medium/modular_transformer_lm_gpt3_medium_LR=0.0005/"
        }
}
