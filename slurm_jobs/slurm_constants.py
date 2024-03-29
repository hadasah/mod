
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
        "MEM_GB": 42,
        "JOBTIME": '48:00:00',
    },
    # Suchin's FAIR account
    "suching": {
        "SECONDARY_DATA_BIN": '/private/home/suching/raw_data/demix_scale/data-bin/',
        "DATA_BIN":'/private/home/suching/raw_data/data-bin-big/',
        "MOD_FOLDER":'/private/home/suching/mod/',
        "MODEL_FOLDER":'/checkpoint/suching/mod_publication/',
        "LOG_FOLDER": '/checkpoint/suching/mod_publication/',
        "JQ_PATH":'jq',
        "SLURM_ACCOUNT": "fairusers",
        "SLURM_PARTITION": "devlab,learnlab",
        "CONDA_ENV": 'mod',
        "NUM_CPUS": 10,
        "MEM_GB": 480,
        "MEM_GB_MOD": 48,
        "JOBTIME": '72:00:00',
    },
    # Margaret's FAIR account
    "margaretli": {
        "SECONDARY_DATA_BIN": '/private/home/suching/raw_data/demix_scale/data-bin/',
        "DATA_BIN":'/private/home/suching/raw_data/data-bin-big/',
        "MOD_FOLDER":'/private/home/margaretli/gitfiles/mod/',
        "MODEL_FOLDER":'/checkpoint/margaretli/mod_publication/',
        "LOG_FOLDER": '/checkpoint/margaretli/mod_publication/',
        "JQ_PATH":'jq',
        "SLURM_ACCOUNT": "fairusers",
        "SLURM_PARTITION": "devlab,learnlab,scavenge",
        "CONDA_ENV": 'mod_new',
        "NUM_CPUS": 10,
        "MEM_GB": 480,
        "MEM_GB_MOD": 48,
        "JOBTIME": '72:00:00',
    }
}
