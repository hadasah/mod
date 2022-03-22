from slurm_jobs.slurm_job import run_grid
from slurm_jobs.slurm_constants_margaret_klone import *
import os

SWEEP_NAME = "sweep_gpt3_small_to_mod"
DEBUG_MODE = True
DRY_MODE = False
name_keys = []
NUM_GPUS = 8

CHECKPOINT_ID = 'best'

grids = {
    SWEEP_NAME: {
        'fixed_args': '',
        'positional_args': {
            "NUM_GPUS": [NUM_GPUS],
            "DISTRIBUTED_PORT": [43212],
            "MODEL": ['transformer_lm_gpt3_small'],
            "EXPERIMENT": ['dense'],
            "DATA_BIN": [DATA_BIN],
            "ROOT_MODEL_FOLDER": [MODEL_FOLDER],
            "NUM_STEPS": [18000, 36000],
            "UPDATE_FREQ": [32],
            "LR": [1e-4, 2e-4, 5e-4, 1e-3, 2e-3],
            "EXPERIMENT_SUFFIX": ["lr_sweep"],
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
        prefix=f'bash {MOD_FOLDER}/demix/downstream_train.sh',
        gpus=NUM_GPUS,
        cpus=4,
        nodes=1,
        #TODO change these
        account='zlab',
        partition='ckpt',
        jobtime='2:00:00',
        mem_gb=40,
        job_id_start=1,
        debug_mode=DEBUG_MODE,
        dry_mode=DRY_MODE,
        add_name='end',
        DIR_PATH=DEMIX_FOLDER,
        
    )
