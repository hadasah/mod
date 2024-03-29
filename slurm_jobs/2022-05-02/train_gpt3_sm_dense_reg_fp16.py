from slurm_jobs.slurm_job import run_grid
from slurm_jobs.slurm_constants import CONSTANTS
import os

username = os.getlogin()
RUN_CONSTANTS = CONSTANTS.get(username)
if RUN_CONSTANTS is None:
    raise Error("username isn't defined in slurm_constants file")
MOD_FOLDER = RUN_CONSTANTS.get('MOD_FOLDER')

SWEEP_NAME = "new_reg_fp16_gpt3_small"
DEBUG_MODE = False
DRY_MODE = False
name_keys = ["EXPERIMENT", "MODEL", "LR", "NUM_STEPS"]
NUM_GPUS = 8

grids = {
    SWEEP_NAME: {
        'fixed_args': '',
        'positional_args': {
            "NUM_GPUS": [NUM_GPUS],
            "MODEL": ['transformer_lm_gpt3_small'],
            "EXPERIMENT": ['dense', 'demix'],
            "DATA_BIN": [RUN_CONSTANTS.get('DATA_BIN')],
            "PARAMS_TO_FREEZE": ["None"],
            "COPYING_MODEL_FOLDER": ["None"],
            "MODEL_FOLDER": [RUN_CONSTANTS.get('MODEL_FOLDER')],
            "SUBFOLDER_NAME": [SWEEP_NAME],
            "PHASE_ONE_RATIO": ["None"],
            "RESET_ITEMS": ["None"],
            "NUM_STEPS": [18000, 36000],
            "UPDATE_FREQ": [8],
            "LR": [1e-3],
            "WANDB_PROJECT": ['test'],
            "WANDB_ENTITY": ['scaling-demix'],
            "MOD_FOLDER": [MOD_FOLDER],
            "DOMAIN_IDS": ['all']
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
        #TODO change these
        account='bdata',
        partition='gpu-rtx6k',
        jobtime='48:00:00',
        mem_gb=50,
        job_id_start=1,
        debug_mode=DEBUG_MODE,
        dry_mode=DRY_MODE,
        add_name='end',
        DIR_PATH=MOD_FOLDER,
    )
