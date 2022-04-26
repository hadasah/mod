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
SWEEP_NAME = "sweep_gpt3_small_to_mod_suchin_dense"
DEBUG_MODE = False
DRY_MODE = False
name_keys = ["MODEL", "DOMAIN_ID", "LOAD_FROM_STEP", "NUM_STEPS"]
NUM_GPUS = 8
NUM_NODES = 1
# CHECKPOINTS_TOP_FOLDER = '/gscratch/zlab/margsli/demix-checkpoints/models'
# NEW_MODEL_TOP_FOLDER = '/gscratch/zlab/margsli/demix-checkpoints/models_test'
CHECKPOINTS_TOP_FOLDER = '/checkpoint/suching/margaret_sweep_rerun/small/'
NEW_MODEL_TOP_FOLDER = '/checkpoint/suching/mod_sweep/_modular_gpt3_small_36K/modular_gpt3_small_36K_LR=0.001/'

re_string = ''
FOLDERS = mod_checkpoint_utils.find_folders(CHECKPOINTS_TOP_FOLDER, re_string=re_string)
print(FOLDERS)

MODEL_DIR='/checkpoint/suching/margaret_sweep_rerun/small/_EXPERIMENT=dense_NUMSTEPS=36000_LR=0.001/'
SERIALIZATION_DIR='/checkpoint/suching/suchin_mod/small/_EXPERIMENT=dense_NUMSTEPS=36000_LR=0.001_mod/'
grids = {
    SWEEP_NAME: {
        'fixed_args': '',
        'positional_args': {
            "DATA_PATH": [RUN_CONSTANTS.get('DATA_BIN')],
            "DOMAIN_ID": [i for i in range(8)],
            "MODEL_DIR": [MODEL_DIR],
            "LOAD_FROM_STEP": [6000,18000,30000],
            "EXPERIMENT": ["full"],
            "SERIALIZATION_DIR": [SERIALIZATION_DIR],
            "FILE_SUFFIX": ["test"],
            "WANDB_PROJECT": ['mod'],
            "MOD_FOLDER": [MOD_FOLDER],
        },
        'named_args': {},
    },
}

# run_grid(
#         grid,
#         name_keys,
#         sweep_name,
#         user=os.environ['USER'],
#         prefix=f'bash {DEMIX_FOLDER}/demix/train.sh',
#         gpus=8,
#         cpus=10,
#         nodes=NUM_NODES,
#         #TODO change these
#         account='fairusers',
#         partition='devlab,learnlab',
#         jobtime='72:00:00',
#         mem_gb=480,
#         job_id_start=1,
#         debug_mode=DEBUG_MODE,
#         dry_mode=DRY_MODE,
#         add_name='end',
#         DIR_PATH=DEMIX_FOLDER,
#         conda_env_name='mod'
#     )

for sweep_name, grid in grids.items():
    run_grid(
        grid,
        name_keys,
        sweep_name,
        user=os.environ['USER'],
        prefix=f'bash {MOD_FOLDER}/demix/mod.sh',
        gpus=NUM_GPUS,
        cpus=10,
        nodes=NUM_NODES,
        #TODO change these
        account=RUN_CONSTANTS.get('SLURM_ACCOUNT'),
        partition=RUN_CONSTANTS.get('SLURM_PARTITION'),
        jobtime='72:00:00',
        mem_gb=480,
        job_id_start=1,
        debug_mode=DEBUG_MODE,
        dry_mode=DRY_MODE,
        add_name='end',
        DIR_PATH=MOD_FOLDER,
        conda_env_name='mod'
        
    )
