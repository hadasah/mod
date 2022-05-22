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
SWEEP_NAME = "diff_pretrain_gpt3_small_to_mod2"
DEBUG_MODE = True
DRY_MODE = False
name_keys = ["MODEL", "CHECKPOINTS_SUBFOLDER", "PHASE_ONE_UPDATE_NUM", "RESET_ITEMS", "LR", "UPDATE_FREQ", "DOMAIN_ID"]
NUM_GPUS = 1
NUM_NODES = 1
CHECKPOINTS_TOP_FOLDER = '/gscratch/zlab/margsli/demix-checkpoints/models/diff_pretrain_gpt3_small/MODEL=transformerlmgpt3small_EXPERIMENT=dense_DOMAINIDS=5_NUMSTEPS=300000_UPDATEFREQ=8_LR=0.0005/'
NEW_MODEL_TOP_FOLDER = '/gscratch/zlab/margsli/demix-checkpoints/models/diff_pretrain_gpt3_small_to_mod2/MODEL=transformerlmgpt3small_EXPERIMENT=dense_NUMSTEPS=300000_UPDATEFREQ=8_LR=0.0005_DOMAINIDS=5/'

re_string = ''
FOLDERS = mod_checkpoint_utils.find_folders(CHECKPOINTS_TOP_FOLDER, re_string=re_string)
print(FOLDERS)

grids = {
    SWEEP_NAME: {
        'fixed_args': '',
        'positional_args': {
            "SWEEP_NAME": [SWEEP_NAME],
            "NUM_GPUS": [NUM_GPUS],
            "MODEL": ['transformer_lm_gpt3_small'],
            "EXPERIMENT": ['mod'],
            "DATA_BIN": [RUN_CONSTANTS.get('DATA_BIN')],
            "DOMAIN_ID": [i for i in range(8)],
            "PARAMS_TO_FREEZE": ["None"],
            "COPYING_MODEL_FOLDER": [CHECKPOINTS_TOP_FOLDER],
            # "COPYING_MODEL_FOLDER": ["None"],
            "NEW_MODEL_TOP_FOLDER": [NEW_MODEL_TOP_FOLDER],
            "CHECKPOINTS_SUBFOLDER": FOLDERS,
            "PHASE_ONE_RATIO": ["None"],
            "PHASE_ONE_UPDATE_NUM": [72000],
            "RESET_ITEMS": ['dataloader,meters,optimizer'],
            "NUM_STEPS": [300000],
            "UPDATE_FREQ": [8],
            "LR": [5e-4],
            "WANDB_PROJECT": ['test'],
            "WANDB_ENTITY": ['scaling-demix'],
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
        cpus=RUN_CONSTANTS.get('NUM_CPUS'),
        nodes=NUM_NODES,
        account='zlab',
        partition=RUN_CONSTANTS.get('SLURM_PARTITION'),
        # partition='ckpt',
        jobtime=RUN_CONSTANTS.get('JOBTIME'),
        # jobtime=':00:00',
        mem_gb=RUN_CONSTANTS.get('MEM_GB'),
        job_id_start=1,
        debug_mode=DEBUG_MODE,
        dry_mode=DRY_MODE,
        add_name='end',
        DIR_PATH=MOD_FOLDER,
        logroot=NEW_MODEL_TOP_FOLDER,
        saveroot=NEW_MODEL_TOP_FOLDER,
        conda_env_name=RUN_CONSTANTS.get('CONDA_ENV_NAME'),
    )
