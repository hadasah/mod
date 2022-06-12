from mod_utils import mod_checkpoint_utils
from slurm_jobs.slurm_constants import CONSTANTS
from slurm_jobs.slurm_job import run_grid
import fairseq
import os
import numpy as np

username = os.getlogin()
if username not in CONSTANTS:
    raise Error("username isn't defined in slurm_constants file")
RUN_CONSTANTS = CONSTANTS.get(username)
MOD_FOLDER = RUN_CONSTANTS.get('MOD_FOLDER')
DEBUG_MODE = True
DRY_MODE = False
name_keys = ["MODEL", "DOMAIN_ID", "NUM_GPUS", "UPDATE_FREQ", "BATCH_SIZE", "LOAD_FROM_STEP", "NUM_STEPS", "LR"]

MODEL = 'transformer_lm_gpt3_small'
SPECS = {"transformer_lm_gpt3_small": {
                # "MODEL_DIR": "/checkpoint/suching/mod_publication/NUMGPUS=16_EXPERIMENT=dense_NUMSTEPS=80000_UPDATEFREQ=32_LR=0.0005",
                "MODEL_DIR": "None",
                "SERIALIZATION_DIR": "/checkpoint/suching/mod_publication/mod/small/PHASE1_16GPU_MOD_2GPU",
                "NUM_GPUS": 2,
                "TOTAL_STEPS": 80000,
            },
            "transformer_lm_gpt3_medium": 32,
            "transformer_lm_gpt3_large": 64,
            "transformer_lm_gpt3_xl": 128
            }[MODEL]

# LOAD_FROM_STEP = [SPECS['TOTAL_STEPS'] - int(SPECS['TOTAL_STEPS'] * i) for i in np.arange(0.1, 1.0, 0.1)]
LOAD_FROM_STEP = [8000]
NUM_NODES = 1
SWEEP_NAME = f"sweep_gpt3_small_mod_" + SPECS['SERIALIZATION_DIR'].split('/')[-1]

# CHECKPOINTS_TOP_FOLDER = '/gscratch/zlab/margsli/demix-checkpoints/models'
# NEW_MODEL_TOP_FOLDER = '/gscratch/zlab/margsli/demix-checkpoints/models_test'
# CHECKPOINTS_TOP_FOLDER = '/checkpoint/suching/margaret_sweep_rerun/small/'
# NEW_MODEL_TOP_FOLDER = f'/checkpoint/suching/suchin_mod_{NUM_GPUS}_GPU/_modular_gpt3_small_36K/modular_gpt3_small_36K_LR=0.001/'

# re_string = ''
# FOLDERS = mod_checkpoint_utils.find_folders(CHECKPOINTS_TOP_FOLDER, re_string=re_string)
# print(FOLDERS)

# MODEL_DIR='/checkpoint/suching/margaret_sweep_rerun/small/_EXPERIMENT=dense_NUMSTEPS=36000_LR=0.001/'
# MODEL_DIR = CHECKPOINTS_TOP_FOLDER + '/NUMGPUS=16_EXPERIMENT=dense_NUMSTEPS=80000_UPDATEFREQ=32_LR=0.0005/'

grids = {
    SWEEP_NAME: {
        'fixed_args': '',
        'positional_args': {
            "DATA_PATH": [RUN_CONSTANTS.get('DATA_BIN')],
            "DOMAIN_ID": [i for i in range(8)],
            "MODEL_DIR": [SPECS['MODEL_DIR']],
            "ARCH": [MODEL],
            "LOAD_FROM_STEP": LOAD_FROM_STEP,
            "EXPERIMENT": ["full"],
            "SERIALIZATION_DIR": [SPECS['SERIALIZATION_DIR']],
            "FILE_SUFFIX": ["test"],
            "TOTAL_STEPS": [SPECS['TOTAL_STEPS']],
            "WANDB_PROJECT": ['mod'],
            "UPDATE_FREQ": [32],
            "LR": [5e-4],
            "NUM_GPUS": [SPECS['NUM_GPUS']],
            "MOD_FOLDER": [MOD_FOLDER],
            "PORT": [np.random.randint(1024, 65535)]
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
        gpus=SPECS['NUM_GPUS'],
        cpus=10,
        nodes=NUM_NODES,
        #TODO change these
        account=RUN_CONSTANTS.get('SLURM_ACCOUNT'),
        partition=RUN_CONSTANTS.get('SLURM_PARTITION'),
        jobtime='48:00:00',
        mem_gb=40,
        job_id_start=1,
        volta=True,
        volta32=False,
        debug_mode=DEBUG_MODE,
        dry_mode=DRY_MODE,
        add_name='end',
        DIR_PATH=MOD_FOLDER,
        conda_env_name='mod'
        
    )
