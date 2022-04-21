from mod_utils import mod_checkpoint_utils
from slurm_constants import CONSTANTS
from slurm_jobs.slurm_job import run_grid
import fairseq
import os

username = os.getlogin()
if username not in CONSTANTS:
    raise Error("username isn't defined in slurm_constants file")
RUN_CONSTANTS = CONSTANTS.get(username)
MOD_FOLDER = RUN_CONSTANTS.get('MOD_FOLDER')

SWEEP_NAME = "sweep_gpt3_medium_to_mod"
DEBUG_MODE = True
DRY_MODE = False
name_keys = ["MODEL", "DOMAIN_ID", "PHASE_ONE_RATIO"]
NUM_GPUS = 1

# CHECKPOINTS_TOP_FOLDER = '/gscratch/zlab/margsli/demix-checkpoints/models'
# NEW_MODEL_TOP_FOLDER = '/gscratch/zlab/margsli/demix-checkpoints/models_test'
CHECKPOINTS_TOP_FOLDER = '/gscratch/zlab/margsli/demix-checkpoints/models/_modular_gpt3_medium_16gpuday/modular_gpt3_medium_16gpuday_DOMAIN=1b_LR=0.001/'
NEW_MODEL_TOP_FOLDER = '/gscratch/zlab/margsli/demix-checkpoints/models_test/_modular_gpt3_medium_16gpuday/modular_gpt3_medium_16gpuday_DOMAIN=1b_LR=0.001/'

re_string = ''
FOLDERS = mod_checkpoint_utils.find_folders(CHECKPOINTS_TOP_FOLDER, re_string=re_string)
print(FOLDERS)

grids = {
    SWEEP_NAME: {
        'fixed_args': '',
        'positional_args': {
            "NUM_GPUS": [NUM_GPUS],
            "MODEL": ['transformer_lm_gpt3_medium'],
            "DATA_BIN": [RUN_CONSTANTS.get('DATA_BIN')],
            "DOMAIN_ID": [i for i in range(1,8)],
            "PARAMS_TO_FREEZE": ["None"],
            "COPYING_MODEL_FOLDER": [CHECKPOINTS_TOP_FOLDER],
            "NEW_MODEL_TOP_FOLDER": [NEW_MODEL_TOP_FOLDER],
            "CHECKPOINTS_SUBFOLDER": FOLDERS,
            "PHASE_ONE_RATIO": [0.5],
            "NUM_STEPS": [18000],
            "UPDATE_FREQ": [32],
            "LR": [1e-3],
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
        prefix=f'bash {MOD_FOLDER}/demix/modular_train.sh',
        gpus=NUM_GPUS,
        cpus=4,
        nodes=1,
        #TODO change these
        account='bdata',
        partition='gpu-rtx6k',
        jobtime='24:00:00',
        mem_gb=42,
        job_id_start=1,
        debug_mode=DEBUG_MODE,
        dry_mode=DRY_MODE,
        add_name='end',
        DIR_PATH=MOD_FOLDER,
        
    )
