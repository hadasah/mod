from slurm_jobs.slurm_job import run_grid
import os
import re
from slurm_jobs.slurm_constants import *
from slurm_jobs.model_specs import EVAL_FOLDERS

DEBUG_MODE = False
DRY_MODE = False
name_keys = []
NUM_GPUS = 1
NUM_NODES = 1
username = os.getlogin()
if username not in CONSTANTS:
        raise Error("username isn't defined in slurm_constants file")
RUN_CONSTANTS = CONSTANTS.get(username)
MOD_FOLDER = RUN_CONSTANTS.get('MOD_FOLDER')
# MODEL_FOLDER = RUN_CONSTANTS.get('MODEL_FOLDER') + "/small/"

# MODEL_FOLDER = "/checkpoint/suching/mod/_modular_gpt3_small_80K/modular_gpt3_small_80K_LR=0.0005/"


MODEL='transformer_lm_gpt3_small'
SWEEP_NAME = f"eval_sweep_{MODEL}_dense"
EVAL_FOLDER = EVAL_FOLDERS[MODEL]
MODEL_FOLDER ='/checkpoint/suching/mod/_modular_gpt3_small_80K/modular_gpt3_small_80K_LR=0.0005/MODEL=transformerlmgpt3small_DOMAINID=6_LOADFROMSTEP=24000_RESETITEMS=dataloader_UPDATEFREQ=32_LR=0.0005/'
MODEL_FOLDER = '/checkpoint/suching/mod/_adaptation_transformer_lm_gpt3_small/adaptation_transformer_lm_gpt3_small_LR=0.0005/MODEL=transformerlmgpt3small_DOMAINID=11_LOADFROMSTEP=-1_RESETITEMS=meters,dataloader,optimizer,lr-scheduler_UPDATEFREQ=32_AVERAGE=True_LR=0.0005/'

DATA_BIN = RUN_CONSTANTS.get('DATA_BIN')
JQ_PATH = RUN_CONSTANTS.get('JQ_PATH')

# This regex looks in MODEL_FOLDER's subfolders for matches
# WANTED_FOLDER_REGEX = 'MODEL=transformerlmgpt3small_DOMAINID=0_PHASEONERATIO=0.6_RESETITEMS=dataloader_UPDATEFREQ=32_LR=0.0005'
WANTED_FOLDER_REGEX='.'
# Used to distinguish between my naming conventions for demix vs modular models
MODEL_TYPE = 'dense'
# Determines where the posteriors and results gets saved 
EVAL_FOLDER_ID = 'Base_dense'
# Comma separated list of the checkpoint IDs. 
#Unfortunately this can't be set per job, I'm assuming we're always setting the right # updates
CHECKPOINT_ID = 'last'

EVAL_SCRIPT = f'{MOD_FOLDER}/demix/mix_eval_pipeline.sh' if MODEL_TYPE in ['demix', 'modular'] else f'{MOD_FOLDER}/demix/eval_pipeline.sh'
all_runs = os.listdir(MODEL_FOLDER)
regex = re.compile(WANTED_FOLDER_REGEX)
selected_folders = [folder for folder in all_runs if regex.match(folder)]

grids = {
    SWEEP_NAME: {
        'fixed_args': '',
        'positional_args': {
            "DATA_BIN": [DATA_BIN],
            "ROOT_MODEL_FOLDER": [MODEL_FOLDER],
            "MODEL_FOLDERS": '.',
            "CHECKPOINT_ID": [CHECKPOINT_ID],
            "SPLIT": ['test'],
            "DOMAIN_ID": [11],
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
        prefix=f'bash {EVAL_SCRIPT}',
        gpus=NUM_GPUS,
        cpus=10,
        nodes=NUM_NODES,
        #TODO change these
        account=RUN_CONSTANTS['SLURM_ACCOUNT'],
        partition=RUN_CONSTANTS['SLURM_PARTITION'],
        jobtime='2:00:00',
        mem_gb=40,
        job_id_start=1,
        debug_mode=DEBUG_MODE,
        dry_mode=DRY_MODE,
        DIR_PATH=MOD_FOLDER,
        conda_env_name='mod',
    )
