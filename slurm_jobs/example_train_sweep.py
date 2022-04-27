from slurm_jobs.slurm_job import run_grid
from slurm_jobs.slurm_constants import CONSTANTS
import os

username = os.getlogin()
RUN_CONSTANTS = CONSTANTS.get(username)
if RUN_CONSTANTS is None:
    raise Error("username isn't defined in slurm_constants file")
MOD_FOLDER = RUN_CONSTANTS.get('MOD_FOLDER')

SWEEP_NAME = "sweep_gpt3_small_64_GPUs"
DEBUG_MODE = True
DRY_MODE = False
name_keys = []
NUM_GPUS = 64
NUM_NODES = NUM_GPUS // 8

grids = {
    SWEEP_NAME: {
        'fixed_args': '',
        'positional_args': {
            "NUM_GPUS": [NUM_GPUS],
            "DISTRIBUTED_PORT": [43212],
            "MODEL": ['transformer_lm_gpt3_small'],
            "EXPERIMENT": ['dense', 'demix'],
            "MODEL_DIR": [RUN_CONSTANTS.get('MODEL_FOLDER')],
            "DATA_BIN": [RUN_CONSTANTS.get('DATA_BIN')],
            "NUM_STEPS": [36000, 55],
            "UPDATE_FREQ": [32],
            "LR": [1e-3, 55],
            "WANDB_PROJECT": ['margaret_sweep'],
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
        jobtime='72:00:00',
        mem_gb=480,
        job_id_start=1,
        debug_mode=DEBUG_MODE,
        dry_mode=DRY_MODE,
        add_name='end',
        DIR_PATH=MOD_FOLDER,
    )
