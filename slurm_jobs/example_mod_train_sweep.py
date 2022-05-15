from mod_utils import mod_checkpoint_utils
from slurm_jobs.slurm_constants import CONSTANTS
from slurm_jobs.slurm_job import run_grid
import fairseq
import os

username = os.getlogin()
if username not in CONSTANTS:
    raise Error("username isn't defined in slurm_constants file")
RUN_CONSTANTS = CONSTANTS.get(username)
MOD_FOLDER = RUN_CONSTANTS.get('MOD_FOLDER')
SWEEP_NAME = "test_gpt3_small_to_mod_margsli"
DEBUG_MODE = False
DRY_MODE = False
name_keys = ["MODEL",  "LOAD_FROM_STEP", "RESET_ITEMS", "LR", "UPDATE_FREQ", "DOMAIN_ID"]
NUM_GPUS = 2
NUM_NODES = 1
# CHECKPOINTS_TOP_FOLDER = '/gscratch/zlab/margsli/demix-checkpoints/models/suchin/'
# NEW_MODEL_TOP_FOLDER = '/gscratch/zlab/margsli/demix-checkpoints/models_test/suchin/'
CHECKPOINTS_TOP_FOLDER = '/checkpoint/suching/fp16/'
NEW_MODEL_TOP_FOLDER = '/checkpoint/suching/mod/_modular_gpt3_small_80K/modular_gpt3_small_80K_LR=0.0005/'

re_string = ''
FOLDERS = mod_checkpoint_utils.find_folders(CHECKPOINTS_TOP_FOLDER, re_string=re_string)
print(FOLDERS)

grids = {
    SWEEP_NAME: {
        'fixed_args': '',
        'positional_args': {
            "NUM_GPUS": [NUM_GPUS],
            "MODEL": ['transformer_lm_gpt3_small'],
            "DATA_BIN": [RUN_CONSTANTS.get('DATA_BIN')],
            "DOMAIN_ID": [i for i in range(8)],
            "PARAMS_TO_FREEZE": ["None"],
            "COPYING_MODEL_FOLDER": [CHECKPOINTS_TOP_FOLDER],
            "NEW_MODEL_TOP_FOLDER": [NEW_MODEL_TOP_FOLDER],
            "CHECKPOINTS_SUBFOLDER": FOLDERS,
            "LOAD_FROM_STEP": [24000],
            "RESET_ITEMS": ['dataloader'],
            "NUM_STEPS": [80000],
            "UPDATE_FREQ": [32],
            "LR": [5e-4],
            "WANDB_PROJECT": ['mod_test'],
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
        logroot=NEW_MODEL_TOP_FOLDER,
        saveroot=NEW_MODEL_TOP_FOLDER,
        conda_env_name=RUN_CONSTANTS.get('CONDA_ENV_NAME'),
    )
