SPECS = {
            "transformer_lm_gpt3_small": {
                'NUM_GPUS': 16,
                "NUM_MOD_GPUS": 2,
                "NUM_STEPS": 80000,
                "MOD_FROM_STEPS": [24000, 56000, 80000],
                "SAVE_INTERVAL_UPDATES": 8000,
                "LR": 5e-4,
                "UF": 32
            },
            "transformer_lm_gpt3_medium": {
                "NUM_GPUS": 32,
                "NUM_MOD_GPUS": 4,
                "NUM_STEPS": 32000,
                "MOD_FROM_STEPS": [9600, 16000, 24000],
                "SAVE_INTERVAL_UPDATES": 2000,
                "LR": 5e-4,
                "UF": 32
            },
            "transformer_lm_gpt3_large": 64,
            "transformer_lm_gpt3_xl": 128
}