from slurm_jobs.slurm_job import run_grid
from slurm_constants_margaret_klone import *
import os
import re

SWEEP_NAME = "eval_sweep_gpt3_medium_demix"
DEBUG_MODE = False
DRY_MODE = True
name_keys = []
NUM_GPUS = 8

# This regex looks in MODEL_FOLDER's subfolders for matches
WANTED_FOLDER_REGEX = '.*demix.*'
# Used to distinguish between my naming conventions for demix vs modular models
MODEL_TYPE = 'demix'
# Determines where the posteriors and results gets saved 
EVAL_FOLDER_ID = 'Base_demix'
# Comma separated list of the checkpoint IDs. 
#Unfortunately this can't be set per job, I'm assuming we're always setting the right # updates
CHECKPOINT_IDS = 'best,best,best,best,best,best,best,best'

EVAL_SCRIPT = f'{MOD_FOLDER}/demix/mix_eval_pipeline.sh' if MODEL_TYPE in ['demix', 'modular'] else f'{MOD_FOLDER}/demix/eval_pipeline.sh'
all_runs = os.listdir(MODEL_FOLDER)
regex = re.compile(WANTED_FOLDER_REGEX)
selected_folders = [folder for folder in all_runs if regex.match(folder)]

grids = {
    SWEEP_NAME: {
        'fixed_args': '',
        'positional_args': {
            "NUM_GPUS": [NUM_GPUS],
            "DATA_BIN": [DATA_BIN],
            "ROOT_MODEL_FOLDER": [MODEL_FOLDER],
            "MODEL_FOLDERS": selected_folders,
            "CHECKPOINT_IDS": [CHECKPOINT_IDS],
            "DOMAIN_ID": [i for i in range(16)],
            "ENSEMBLE_TYPE": ['cached_prior'],
            "MODEL_TYPE": [MODEL_TYPE],
            "GENERALIST_MODEL": ["None"],
            "TOP_K": [8],
            "EVAL_FOLDER_ID": [EVAL_FOLDER_ID],
            "EXCLUDE_EXPERT": ["False"],
            "ONLY_USE_DOMAIN_EXPERT": ['False'],
            "MOD_FOLDER": [MOD_FOLDER],
            "JQ_PATH": [JQ_PATH]
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
        DIR_PATH=MOD_FOLDER,
        #TODO change this
        conda_env_name='latest',
    )
