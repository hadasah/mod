from mod_utils import mod_checkpoint_utils
from slurm_jobs.slurm_constants import CONSTANTS
from slurm_jobs.slurm_job import run_grid
from slurm_jobs.model_specs import SPECS
import fairseq
import os
import numpy as np

username = os.getlogin()
if username not in CONSTANTS:
    raise Error("username isn't defined in slurm_constants file")
RUN_CONSTANTS = CONSTANTS.get(username)
MOD_FOLDER = RUN_CONSTANTS.get('MOD_FOLDER')
DEBUG_MODE = True
DRY_MODE = False
name_keys = ["MODEL", "DOMAIN_ID", "NUM_GPUS", "UPDATE_FREQ", "BATCH_SIZE", "LOAD_FROM_STEP", "NUM_STEPS", "LR"]

MODEL = 'transformer_lm_gpt3_medium'

SPECS = SPECS[MODEL]
NUM_GPUS = SPECS['NUM_MOD_GPUS']
NUM_NODES = 1 if NUM_GPUS < 8 else NUM_GPUS // 8

SWEEP_NAME = f"sweep_{MODEL}_mod_from_scratch"

grids = {
    SWEEP_NAME: {
        'fixed_args': '',
        'positional_args': {
            "DATA_PATH": [RUN_CONSTANTS.get('DATA_BIN')],
            "DOMAIN_ID": [i for i in range(8)],
            "MODEL_DIR": ["None"],
            "ARCH": [MODEL],
            "LOAD_FROM_STEP": ["None"],
            "EXPERIMENT": ["full"],
            "SERIALIZATION_DIR": [f"/checkpoint/suching/mod_publication/mod/{MODEL}_MOD_{NUM_GPUS}_GPU"],
            "FILE_SUFFIX": ["test"],
            "TOTAL_STEPS": [SPECS['TOTAL_STEPS']],
            "WANDB_PROJECT": ['mod'],
            "UPDATE_FREQ": [SPECS['UF']],
            "LR": [SPECS['LR']],
            "NUM_GPUS": [NUM_GPUS],
            "MOD_FOLDER": [MOD_FOLDER],
            "PORT": [np.random.randint(1024, 65535)]
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
        prefix=f'bash {MOD_FOLDER}/demix/mod.sh',
        gpus=SPECS['NUM_GPUS'],
        cpus=RUN_CONSTANTS.get('NUM_CPUS'),
        nodes=NUM_NODES,
        #TODO change these
        account=RUN_CONSTANTS.get('SLURM_ACCOUNT'),
        partition=RUN_CONSTANTS.get('SLURM_PARTITION'),
        jobtime=RUN_CONSTANTS.get('JOBTIME'),
        mem_gb=40,
        job_id_start=1,
        volta=True,
        volta32=False,
        debug_mode=DEBUG_MODE,
        dry_mode=DRY_MODE,
        add_name='end',
        DIR_PATH=MOD_FOLDER,
        conda_env_name='mod'
        
    )
