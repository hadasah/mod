from slurm_jobs.slurm_job import run_grid
from slurm_jobs.slurm_constants import *
import os
import re

username = os.getlogin()
if username not in CONSTANTS:
    raise Error("username isn't defined in slurm_constants file")
RUN_CONSTANTS = CONSTANTS.get(username)
MOD_FOLDER = RUN_CONSTANTS.get('MOD_FOLDER')
MODEL_FOLDER = RUN_CONSTANTS.get('MODEL_FOLDER')
DATA_BIN = RUN_CONSTANTS.get('DATA_BIN')
JQ_PATH = RUN_CONSTANTS.get('JQ_PATH')

SWEEP_NAME = "eval_diff_pretrain_gpt3_small"
DEBUG_MODE = False
DRY_MODE = False
name_keys = []
NUM_GPUS = 8

# # This regex looks in MODEL_FOLDER's subfolders for matches
WANTED_FOLDER_REGEX = '.*DOMAINIDS=2.*'
SUBFOLDER = 'diff_pretrain_gpt3_small_to_mod'
# # Used to distinguish between my naming conventions for demix vs modular models
MODEL_TYPE = 'modular'

EVAL_SCRIPT = f'{MOD_FOLDER}/demix/mix_eval_pipeline.sh' if MODEL_TYPE in ['demix', 'modular'] else f'{MOD_FOLDER}/demix/eval_pipeline.sh'

MODEL_FOLDER = f'{MODEL_FOLDER}/{SUBFOLDER}'

# Determines where the posteriors and results gets saved 
EVAL_FOLDER_ID = WANTED_FOLDER_REGEX.replace('.*', '')
# Comma separated list of the checkpoint IDs. 
#Unfortunately this can't be set per job, I'm assuming we're always setting the right # updates
CHECKPOINT_IDS = ['2_66000','1_66000','1_66000','1_66000','1_66000','1_66000','1_66000','1_66000']
CHECKPOINT_IDS = ':'.join(CHECKPOINT_IDS)

all_runs = os.listdir(MODEL_FOLDER)
regex = re.compile(WANTED_FOLDER_REGEX)
# selected_folders = [folder for folder in all_runs if regex.match(folder) if "demix" in folder]
selected_folders = ':'.join(sorted([folder for folder in all_runs if regex.match(folder)]))
grids = {
    SWEEP_NAME: {
        'fixed_args': '',
        'positional_args': {
            "NUM_GPUS": [NUM_GPUS],
            "DATA_BIN": [DATA_BIN],
            "ROOT_MODEL_FOLDER": [MODEL_FOLDER],
            "MODEL_FOLDERS": [selected_folders],
            "CHECKPOINT_IDS": [CHECKPOINT_IDS],
            # "DOMAIN_ID": [i for i in range(13, 13)],
            "DOMAIN_ID": [13],
            "ENSEMBLE_TYPE": ['cached_prior'],
            "MODEL_TYPE": [MODEL_TYPE],
            # "GENERALIST_MODEL": ["/checkpoint/suching/margaret_sweep_rerun/small/_EXPERIMENT=dense_NUMSTEPS=36000_LR=0.001/checkpoint_1_30000.pt"],
            "GENERALIST_MODEL": ["None"],
            "TOP_K": [8],
            "EVAL_FOLDER_ID": [EVAL_FOLDER_ID],
            "NUM_STEPS": [6000],
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
        # cpus=RUN_CONSTANTS.get('NUM_CPUS'),
        cpus=4,
        nodes=1,
        #TODO change these
        # account=RUN_CONSTANTS.get('SLURM_ACCOUNT'),
        account='bdata',
        # partition=RUN_CONSTANTS.get('SLURM_PARTITION'),
        partition='gpu-2080ti',
        jobtime='2:00:00',
        # mem_gb=RUN_CONSTANTS.get('MEM_GB'),
        mem_gb=200,
        job_id_start=1,
        debug_mode=DEBUG_MODE,
        dry_mode=DRY_MODE,
        DIR_PATH=MOD_FOLDER,
        #TODO change this
        conda_env_name='latest',
    )
