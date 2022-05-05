from slurm_jobs.slurm_job import run_grid
from slurm_jobs.slurm_constants import *
import os
import re

username = os.getlogin()
if username not in CONSTANTS:
    raise Error("username isn't defined in slurm_constants file")
RUN_CONSTANTS = CONSTANTS.get(username)
MOD_FOLDER = RUN_CONSTANTS.get('MOD_FOLDER')
MODEL_FOLDER = "/checkpoint/suching/mod_sweep/"
DATA_BIN = RUN_CONSTANTS.get('DATA_BIN')
JQ_PATH = RUN_CONSTANTS.get('JQ_PATH')

SWEEP_NAME = "eval_sweep_gpt3_small_mod_8GPU_PHASE1_64GPU_RESET"
DEBUG_MODE = False
DRY_MODE = False
name_keys = []
NUM_GPUS = 8

# /checkpoint/suching/suchin_mod_8_GPUs/small/_EXPERIMENT=dense_NUMSTEPS=36000_LR=0.001/_DOMAIN_5_MOD_STEPS_30000_PHASE1_DENSE
# /checkpoint/suching/mod_sweep/_modular_gpt3_small_36K/modular_gpt3_small_36K_LR=0.001/MODEL=transformerlmgpt3small_DOMAINID=7_PHASEONERATIO=0.25_RESETITEMS=dataloader,meters_UPDATEFREQ=32_LR=0.001

# MODEL=transformerlmgpt3small_DOMAINID=7_PHASEONERATIO=0.25_RESETITEMS=dataloader,meters_UPDATEFREQ=32_LR=0.001

# This regex looks in MODEL_FOLDER's subfolders for matches
WANTED_FOLDER_REGEX = '.*'
# Used to distinguish between my naming conventions for demix vs modular models
MODEL_TYPE = 'modular'
# Determines where the posteriors and results gets saved 
EVAL_FOLDER_ID = 'Base_dense_PHASE_RATIO_0.50'
# Comma separated list of the checkpoint IDs. 
#Unfortunately this can't be set per job, I'm assuming we're always setting the right # updates
CHECKPOINT_IDS = 'best,best,best,best,best,best,best,best'
EVAL_SCRIPT = f'{MOD_FOLDER}/demix/mix_eval_pipeline.sh' if MODEL_TYPE in ['demix', 'modular'] else f'{MOD_FOLDER}/demix/eval_pipeline.sh'
all_runs = os.listdir(MODEL_FOLDER + "/_modular_gpt3_small_36K/")
regex = re.compile(WANTED_FOLDER_REGEX)
print(all_runs)
selected_folders = [folder for folder in all_runs if regex.match(folder)]
print(selected_folders)
grids = {
    SWEEP_NAME: {
        'fixed_args': '',
        'positional_args': {
            "NUM_GPUS": [NUM_GPUS],
            "DATA_BIN": [DATA_BIN],
            "ROOT_MODEL_FOLDER": [MODEL_FOLDER + "/_modular_gpt3_small_36K/"],
            "MODEL_FOLDERS": selected_folders,
            "CHECKPOINT_IDS": [CHECKPOINT_IDS],
            "DOMAIN_ID": [i for i in range(16)],
            "ENSEMBLE_TYPE": ['cached_prior'],
            "MODEL_TYPE": [MODEL_TYPE],
            # "GENERALIST_MODEL": ["/checkpoint/suching/margaret_sweep_rerun/small/_EXPERIMENT=dense_NUMSTEPS=36000_LR=0.001/checkpoint_1_30000.pt"],
            "GENERALIST_MODEL": ["None"],
            "TOP_K": [8],
            "EVAL_FOLDER_ID": [EVAL_FOLDER_ID],
            "PHASE_RATIO": [0.5],
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
        cpus=10,
        nodes=1,
        #TODO change these
        account=RUN_CONSTANTS.get('SLURM_ACCOUNT'),
        partition=RUN_CONSTANTS.get('SLURM_PARTITION'),
        jobtime='2:00:00',
        mem_gb=480,
        job_id_start=1,
        debug_mode=DEBUG_MODE,
        dry_mode=DRY_MODE,
        DIR_PATH=MOD_FOLDER,
        #TODO change this
        conda_env_name='latest',
    )
