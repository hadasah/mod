CONSTANTS = {
    # Margaret's klone account
    "margsli": {
        "DATA_BIN":'/gscratch/zlab/margsli/gitfiles/demix-data/data-bin',
        "MOD_FOLDER":'/gscratch/zlab/margsli/gitfiles/mod',
        "MODEL_FOLDER":'/gscratch/zlab/margsli/demix-checkpoints/models/',
        "LOG_FOLDER": '/gscratch/zlab/margsli/demix-checkpoints/models/',
        "JQ_PATH":'~/jq-linux64',
    },
    # Suchin's FAIR account
    "suching": {
        "DATA_BIN":'/private/home/suching/raw_data/data-bin-big/',
        "MOD_FOLDER":'/private/home/suching/mod/',
        "MODEL_FOLDER":'/checkpoint/suching/suchin_mod/',
        "LOG_FOLDER": '/checkpoint/suching/suchin_mod/',
        "JQ_PATH":'jq',
        "SLURM_ACCOUNT": "fairusers",
        "SLURM_PARTITION": "devlab,learnlab"
    }
}
