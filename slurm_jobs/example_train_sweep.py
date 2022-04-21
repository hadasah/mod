from slurm_jobs.slurm_job import run_grid
import os

SWEEP_NAME = "sweep_gpt3_small"
DEBUG_MODE = True
DRY_MODE = False
name_keys = []
NUM_GPUS = 8
NUM_NODES = NUM_GPUS // 8
#TODO change this
DATA_BIN='/private/home/suching/raw_data/data-bin-big/' 
#TODO change this
DEMIX_FOLDER='/private/home/suching/demix/'
#TODO change this
MODEL_FOLDER = '/checkpoint/suching/margaret_sweep_rerun/'

grids = {
    #"sweep_gpt3_medium": {
    #    'fixed_args': '',
    #    'positional_args': {
    #        "NUM_GPUS": [NUM_GPUS],
    #        "DISTRIBUTED_PORT": [43212],
    #        "MODEL": ['transformer_lm_gpt3_medium'],
    #        "EXPERIMENT": ['dense', 'demix'],
    #        "DATA_BIN": [DATA_BIN],
    #        "ROOT_MODEL_FOLDER": [MODEL_FOLDER + "/medium/"],
    #        "NUM_STEPS": [48000, 55],
    #        "UPDATE_FREQ": [32],
    #        "LR": [5e-4, 2e-3],
    #        "EXPERIMENT_SUFFIX": ["lr_sweep"],
    #        "DEMIX_FOLDER": [DEMIX_FOLDER],
    #    },
    #    'named_args': {},
    #},
    "sweep_gpt3_small": {
        'fixed_args': '',
        'positional_args': {
            "NUM_GPUS": [NUM_GPUS],
            "DISTRIBUTED_PORT": [43212],
            "MODEL": ['transformer_lm_gpt3_small'],
            "EXPERIMENT": ['demix', 'dense'],
            "DATA_BIN": [DATA_BIN],
            "ROOT_MODEL_FOLDER": [MODEL_FOLDER + "/small/"],
            "NUM_STEPS": [36000,55],
            "UPDATE_FREQ": [32],
            "LR": [1e-3, 55],
            "EXPERIMENT_SUFFIX": ["lr_sweep"],
            "DEMIX_FOLDER": [DEMIX_FOLDER],
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
        prefix=f'bash {DEMIX_FOLDER}/demix/train.sh',
        gpus=8,
        cpus=10,
        nodes=NUM_NODES,
        #TODO change these
        account='fairusers',
        partition='devlab,learnlab',
        jobtime='72:00:00',
        mem_gb=480,
        job_id_start=1,
        debug_mode=DEBUG_MODE,
        dry_mode=DRY_MODE,
        add_name='end',
        DIR_PATH=DEMIX_FOLDER,
        conda_env_name='mod'
    )
