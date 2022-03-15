from slurm_jobs.slurm_job import run_grid
import os
import re

SWEEP_NAME = "eval_sweep_gpt3_medium_dense"
DEBUG_MODE = False
DRY_MODE = True
name_keys = []
NUM_GPUS = 1

#TODO change this
DATA_BIN = '/gscratch/zlab/margsli/gitfiles/demix-data/data-bin' 
#TODO change this
MOD_FOLDER = '/gscratch/zlab/margsli/gitfiles/mod'
#TODO change this
# Top level folder for the models -- looks below this for subfolders that contain checkpoints
MODEL_FOLDER = '/gscratch/zlab/margsli/demix-checkpoints/models/'
# MODEL_FOLDER = '/checkpoint/suching/margaret_sweep/medium/'
# This regex looks in MODEL_FOLDER's subfolders for matches
WANTED_FOLDER_REGEX = '.*dense.*'
# Used to distinguish between my naming conventions for demix vs modular models
MODEL_TYPE = 'dense'
# Determines where the posteriors and results gets saved 
EVAL_FOLDER_ID = 'Base_demix'
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
            "DOMAIN_ID": [i for i in range(16)],
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
        account='zlab',
        partition='ckpt',
        jobtime='2:00:00',
        mem_gb=40,
        job_id_start=1,
        debug_mode=DEBUG_MODE,
        dry_mode=DRY_MODE,
        # add_name='end',
        DIR_PATH=MOD_FOLDER,
        
    )
