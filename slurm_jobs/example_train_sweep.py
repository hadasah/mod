from slurm_jobs.slurm_job import run_grid
from slurm_jobs.slurm_constants import CONSTANTS
import os
import numpy as np


username = os.getlogin()
RUN_CONSTANTS = CONSTANTS.get(username)
if RUN_CONSTANTS is None:
    raise Error("username isn't defined in slurm_constants file")
MOD_FOLDER = RUN_CONSTANTS.get('MOD_FOLDER')

MODEL = 'transformer_lm_gpt3_medium'

SPECS = {
            "transformer_lm_gpt3_small": {
                "NUM_GPUS": 16,
                "NUM_STEPS": 72000,
                "SAVE_INTERVAL_UPDATES": 6000,
                "LR": 5e-4,
                "UF": 32
            },
            "transformer_lm_gpt3_medium": {
                "NUM_GPUS": 32,
                "NUM_STEPS": 32000,
                "SAVE_INTERVAL_UPDATES": 2000,
                "LR": 5e-4,
                "UF": 32
            },
            "transformer_lm_gpt3_large": 64,
            "transformer_lm_gpt3_xl": 128
            }[MODEL]
NUM_NODES = SPECS['NUM_GPUS'] // 8
SWEEP_NAME = f"sweep_{MODEL}_{SPECS['NUM_GPUS']}_GPUs"
DEBUG_MODE = True
DRY_MODE = False
name_keys = ["EXPERIMENT", "NUM_STEPS", "UPDATE_FREQ", "LR", "NUM_GPUS"]

grids = {
    SWEEP_NAME: {
        'fixed_args': '',
        'positional_args': {
            "NUM_GPUS": [SPECS['NUM_GPUS']],
            "DISTRIBUTED_PORT": [np.random.randint(1024, 65535)],
            "MODEL": [MODEL],
            "EXPERIMENT": ['dense'],
            "MODEL_DIR": [RUN_CONSTANTS.get('MODEL_FOLDER')],
            "DATA_BIN": [RUN_CONSTANTS.get('DATA_BIN')],
            "NUM_STEPS": [SPECS['NUM_STEPS']],
            "SAVE_INTERVAL_UPDATES": [SPECS['SAVE_INTERVAL_UPDATES']],
            "UPDATE_FREQ": [SPECS["UF"]],
            "LR": [SPECS["LR"]],
            "WANDB_PROJECT": ['publication-test'],
            "WANDB_ENTITY": ['scaling-demix'],
            "MOD_FOLDER": [MOD_FOLDER],
            "ID": [""]
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
        gpus=8,
        cpus=10,
        nodes=NUM_NODES,
        #TODO change these
        account=RUN_CONSTANTS['SLURM_ACCOUNT'],
        partition=RUN_CONSTANTS['SLURM_PARTITION'],
        jobtime='50:00:00',
        mem_gb=480,
        job_id_start=1,
        debug_mode=DEBUG_MODE,
        dry_mode=DRY_MODE,
        add_name='end',
        DIR_PATH=MOD_FOLDER,
    )
