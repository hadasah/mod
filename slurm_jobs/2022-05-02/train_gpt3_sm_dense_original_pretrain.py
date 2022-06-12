from slurm_jobs.slurm_job import run_grid
from slurm_jobs.slurm_constants import CONSTANTS
import os
import numpy as np

username = os.getlogin()
RUN_CONSTANTS = CONSTANTS.get(username)
if RUN_CONSTANTS is None:
    raise Error("username isn't defined in slurm_constants file")
MOD_FOLDER = RUN_CONSTANTS.get('MOD_FOLDER')

SWEEP_NAME = "diff_pretrain_gpt3_small"
DEBUG_MODE = False
DRY_MODE = False
name_keys = ["EXPERIMENT", "MODEL", "LR", "NUM_STEPS", "UPDATE_FREQ", 'DOMAIN_IDS']
NUM_GPUS = 8

grids = {
    SWEEP_NAME: {
        'fixed_args': '',
        'positional_args': {
            "SWEEP_NAME": [SWEEP_NAME],
            "NUM_GPUS": [NUM_GPUS],
            "MODEL": ['transformer_lm_gpt3_small'],
            "EXPERIMENT": ['dense', ],
            "DATA_BIN": [RUN_CONSTANTS.get('DATA_BIN')],
            "DOMAIN_IDS": [7],
            "PARAMS_TO_FREEZE": ["None"],
            "COPYING_MODEL_FOLDER": ["None"],
            "MODEL_FOLDER": [RUN_CONSTANTS.get('MODEL_FOLDER')],
            "SUBFOLDER_NAME": [SWEEP_NAME],
            "PHASE_ONE_RATIO": ["None"],
            "PHASE_ONE_UPDATE_NUM": ["None"],
            "RESET_ITEMS": ["None"],
            "NUM_STEPS": [300000],
            "UPDATE_FREQ": [8],
            "LR": [5e-4],
            "SAVE_INTERVAL_UPDATES": [6000],
            "DISTRIBUTED_PORT": [np.random.randint(1024, 65535)],
            "WANDB_PROJECT": ['test'],
            "WANDB_ENTITY": ['scaling-demix'],
            "MOD_FOLDER": [MOD_FOLDER],
        },
        'named_args': {},
    },
}

for sweep_name, grid in grids.items():
    run_grid(
        grid,
        name_keys,
        sweep_name,
        user=os.environ['USER'],
        prefix=f'bash {MOD_FOLDER}/demix/train.sh',
        gpus=NUM_GPUS,
        cpus=4,
        nodes=1,
        logroot=RUN_CONSTANTS.get('LOG_FOLDER'),
        #TODO change these
        account='zlab',
        partition='gpu-rtx6k',
        jobtime='48:00:00',
        mem_gb=50,
        job_id_start=1,
        debug_mode=DEBUG_MODE,
        dry_mode=DRY_MODE,
        add_name='end',
        DIR_PATH=MOD_FOLDER,
        conda_env_name=RUN_CONSTANTS.get('CONDA_ENV'),
    )
