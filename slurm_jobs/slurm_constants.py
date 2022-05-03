CONSTANTS = {
    # Margaret's klone account
    "margsli": {
        "DATA_BIN":'/gscratch/zlab/margsli/gitfiles/demix-data/data-bin',
        "MOD_FOLDER":'/gscratch/zlab/margsli/gitfiles/mod',
        "MODEL_FOLDER":'/gscratch/zlab/margsli/demix-checkpoints/models/',
        "LOG_FOLDER": '/gscratch/zlab/margsli/demix-checkpoints/models/',
        "JQ_PATH":'~/jq-linux64',
        "SLURM_ACCOUNT": "zlab",
        "SLURM_PARTITION": "gpu-rtx6k",
        "CONDA_ENV": "latest",
        "NUM_CPUS": 5,
        "MEM_GB": 50,
        "JOBTIME": '48:00:00',
    },
    # Suchin's FAIR account
    "suching": {
        "DATA_BIN":'/private/home/suching/raw_data/data-bin-big/',
        "MOD_FOLDER":'/private/home/suching/mod/',
        "MODEL_FOLDER":'/checkpoint/suching/suchin_mod_8_GPU/',
        "LOG_FOLDER": '/checkpoint/suching/suchin_mod_8_GPU/',
        "JQ_PATH":'jq',
        "SLURM_ACCOUNT": "fairusers",
        "SLURM_PARTITION": "devlab,learnlab",
        "CONDA_ENV": 'mod',
        "NUM_CPUS": 10,
        "MEM_GB": 480,
        "JOBTIME": '72:00:00',
    }
}
