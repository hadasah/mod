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

SWEEP_NAME = "sweep_gpt3_small_to_mod"
DEBUG_MODE = False
DRY_MODE = True
name_keys = []
NUM_GPUS = 1

# CHECKPOINTS_TOP_FOLDER = '/checkpoint/suching/margaret_sweep/small/'
# NEW_MODEL_TOP_FOLDER = '/checkpoint/suching/margaret_sweep/small_to_mod/'

CHECKPOINTS_TOP_FOLDER = '/gscratch/zlab/margsli/demix-checkpoints/models'
NEW_MODEL_TOP_FOLDER = '/gscratch/zlab/margsli/demix-checkpoints/models_test'
phase_one_ratio = 0.5

FOLDERS = mod_checkpoint_utils.find_folders(CHECKPOINTS_TOP_FOLDER)
print(FOLDERS)

grids = {
    SWEEP_NAME: {
        'fixed_args': '',
        'positional_args': {
            "NUM_GPUS": [NUM_GPUS],
            "MODEL": ['transformer_lm_gpt3_small'],
            "DATA_BIN": [RUN_CONSTANTS.get('DATA_BIN')],
            "DOMAIN_IDS": [i for i in range(8)],
            "EXPERIMENT_SUFFIX": ["lr_sweep"],
            "PARAMS_TO_FREEZE": ["None"],
            "CHECKPOINTS_TOP_FOLDER": [CHECKPOINTS_TOP_FOLDER],
            "NEW_MODEL_TOP_FOLDER": [NEW_MODEL_TOP_FOLDER],
            "CHECKPOINTS_SUBFOLDER": [FOLDERS],
            "PHASE_ONE_RATIO": [],
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
        account='zlab',
        partition='ckpt',
        jobtime='2:00:00',
        mem_gb=40,
        job_id_start=1,
        debug_mode=DEBUG_MODE,
        dry_mode=DRY_MODE,
        add_name='end',
        DIR_PATH=MOD_FOLDER,
        
    )
