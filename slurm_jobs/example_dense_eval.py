from slurm_jobs.slurm_job import run_grid
import os
import re

SWEEP_NAME = "eval_sweep_gpt3_medium_dense"
DEBUG_MODE = False
DRY_MODE = True
name_keys = []
NUM_GPUS = 8

#TODO change this
DATA_BIN = '/private/home/suching/raw_data/demix_scale/data-bin/' 
#TODO change this
MOD_FOLDER = '/private/home/suching/mod/'
#TODO change this
# Top level folder for the models -- looks below this for subfolders that contain checkpoints
MODEL_FOLDER = '/checkpoint/suching/margaret_sweep_16_GPUs/medium/'
# MODEL_FOLDER = '/checkpoint/suching/margaret_sweep/medium/'
# This regex looks in MODEL_FOLDER's subfolders for matches
MODEL_TYPE_REGEX="dense"
LR_REGEX='0.0005'
NUM_STEPS_REGEX="48000"

DOMAIN_ID = [i for i in range(305)]

WANTED_FOLDER_REGEX=f".*EXPERIMENT={MODEL_TYPE_REGEX}.*NUMSTEPS={NUM_STEPS_REGEX}.*LR={LR_REGEX}.*"
# Used to distinguish between my naming conventions for demix vs modular models
MODEL_TYPE = 'dense'
# Determines where the posteriors and results gets saved 
EVAL_FOLDER_ID = 'Base_dense'
# Comma separated list of the checkpoint IDs. 
#Unfortunately this can't be set per job, I'm assuming we're always setting the right # updates
CHECKPOINT_ID = 'best'

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
            "MODEL_FOLDERS": selected_folders,
            "CHECKPOINT_ID": [CHECKPOINT_ID],
            "SPLIT": ['test'],
            "DOMAIN_ID": DOMAIN_ID,
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
        cpus=4,
        nodes=1,
        #TODO change these
        account='fairusers',
        partition='devlab,learnlab',
        jobtime='2:00:00',
        mem_gb=40,
        job_id_start=1,
        debug_mode=DEBUG_MODE,
        dry_mode=DRY_MODE,
        DIR_PATH=MOD_FOLDER,
        conda_env_name='latest',
    )
