from slurm_jobs.slurm_job import run_grid
from slurm_jobs.slurm_constants import CONSTANTS
import os

username = os.getlogin()
RUN_CONSTANTS = CONSTANTS.get(username)
if RUN_CONSTANTS is None:
    raise Error("username isn't defined in slurm_constants file")
MOD_FOLDER = RUN_CONSTANTS.get('MOD_FOLDER')

SWEEP_NAME = "sweep_gpt3_small_test"
DEBUG_MODE = True
DRY_MODE = False
name_keys = []
NUM_GPUS = 8
NUM_NODES = 1

grids = {
    SWEEP_NAME: {
        'fixed_args': '',
        'positional_args': {
            "SWEEP_NAME": [SWEEP_NAME],
            "NUM_GPUS": [NUM_GPUS],
            "MODEL": ['transformer_lm_gpt3_small'],
            # "EXPERIMENT": ['dense', 'demix'],
            "EXPERIMENT": ['demix'],
            "DATA_PATH": [RUN_CONSTANTS.get('DATA_BIN')],
            "DOMAIN_IDS": ["0,1,2,3"],
            "PARAMS_TO_FREEZE": ["None"],
            "COPYING_MODEL_FOLDER": ["None"],
            "MODEL_FOLDER": [RUN_CONSTANTS.get('MODEL_FOLDER')],
            "SUBFOLDER_NAME": ["None"],
            "PHASE_ONE_RATIO": ["None"],
            "PHASE_ONE_UPDATE_NUM": ["None"],
            "RESET_ITEMS": ["None"],
            "NUM_STEPS": [18000, 36000],
            "UPDATE_FREQ": [32],
            # "LR": [1e-4, 2e-4, 5e-4, 1e-3, 2e-3],
            "LR": [1e-5],
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
        cpus=RUN_CONSTANTS.get('NUM_CPUS'),
        nodes=NUM_NODES,
        account=RUN_CONSTANTS.get('SLURM_ACCOUNT'),
        partition=RUN_CONSTANTS.get('SLURM_PARTITION'),
        jobtime=RUN_CONSTANTS.get('JOBTIME'),
        mem_gb=RUN_CONSTANTS.get('MEM_GB'),
        job_id_start=1,
        debug_mode=DEBUG_MODE,
        dry_mode=DRY_MODE,
        add_name='end',
        DIR_PATH=MOD_FOLDER,
    )
