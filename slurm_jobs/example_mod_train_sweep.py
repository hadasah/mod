from mod_utils import mod_checkpoint_utils
from slurm_jobs.slurm_constants import CONSTANTS
from slurm_jobs.slurm_job import run_grid
import fairseq
import os
import argparse


def main(model, debug=False, dry_mode=False, from_scratch=False, domains=None):
    DEBUG_MODE = debug
    DRY_MODE = dry_mode
    FROM_SCRATCH = from_scratch
    MODEL = model
    username = os.getlogin()
    if username not in CONSTANTS:
        raise Error("username isn't defined in slurm_constants file")
    RUN_CONSTANTS = CONSTANTS.get(username)
    MOD_FOLDER = RUN_CONSTANTS.get('MOD_FOLDER')

    
    SWEEP_NAME = f"sweep_{MODEL}_mod"


    name_keys = ["MODEL",  "LOAD_FROM_STEP", "RESET_ITEMS", "LR", "UPDATE_FREQ", "DOMAIN_ID"]
    from slurm_jobs.model_specs import SPECS
    SPECS = SPECS[MODEL]
    NUM_GPUS = SPECS['NUM_MOD_GPUS']

    NUM_NODES = 1 if NUM_GPUS < 8 else NUM_GPUS // 8


    if FROM_SCRATCH:
        PATH_TO_DENSE_CHECKPOINTS = "None"
        NEW_MODEL_TOP_FOLDER = f'/checkpoint/suching/mod/_modular_{MODEL}/modular_{MODEL}_LR={SPECS["LR"]}_from_scratch/'
        SWEEP_NAME += "_from_scratch"
        FOLDERS = ["None"]
        SPECS['MOD_FROM_STEPS'] = [0]
    else:
        # modify to path to dense checkpoint you want to use
        PATH_TO_DENSE_CHECKPOINTS = '/checkpoint/suching/fp16/'
        NEW_MODEL_TOP_FOLDER = f'/checkpoint/suching/mod/_modular_{MODEL}/modular_{MODEL}_LR={SPECS["LR"]}/'

        re_string = ''
        FOLDERS = mod_checkpoint_utils.find_folders(PATH_TO_DENSE_CHECKPOINTS, re_string=re_string)
        print(FOLDERS)
    if not domains:
        DOMAINS = [i for i in range(8)]
    else:
        DOMAINS = domains
    grids = {
        SWEEP_NAME: {
            'fixed_args': '',
            'positional_args': {
                "NUM_GPUS": [NUM_GPUS],
                "MODEL": [MODEL],
                "DATA_BIN": [RUN_CONSTANTS.get('DATA_BIN')],
                "DOMAIN_ID": DOMAINS,
                "PARAMS_TO_FREEZE": ["None"],
                "COPYING_MODEL_FOLDER": [PATH_TO_DENSE_CHECKPOINTS],
                "NEW_MODEL_TOP_FOLDER": [NEW_MODEL_TOP_FOLDER],
                "CHECKPOINTS_SUBFOLDER": [FOLDERS],
                "LOAD_FROM_STEP": SPECS['MOD_FROM_STEPS'],
                "RESET_ITEMS": ['dataloader'],
                "NUM_STEPS": [SPECS['NUM_STEPS']],
                "UPDATE_FREQ": [SPECS['UF']],
                "LR": [SPECS['LR']],
                "WANDB_PROJECT": ['mod_test'],
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
            account=RUN_CONSTANTS.get('SLURM_ACCOUNT'),
            partition=RUN_CONSTANTS.get('SLURM_PARTITION'),
            jobtime=RUN_CONSTANTS.get('JOBTIME'),
            mem_gb=RUN_CONSTANTS.get('MEM_GB_MOD'),
            job_id_start=1,
            debug_mode=DEBUG_MODE,
            dry_mode=DRY_MODE,
            add_name='end',
            DIR_PATH=MOD_FOLDER,
            logroot=NEW_MODEL_TOP_FOLDER,
            saveroot=NEW_MODEL_TOP_FOLDER,
            conda_env_name=RUN_CONSTANTS.get('CONDA_ENV_NAME'),
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dry-mode', action='store_true')
    parser.add_argument('--model', type=str)
    parser.add_argument('--domains', type=int, nargs="+")
    parser.add_argument('--from-scratch', action='store_true')
    args = parser.parse_args()
    main(args.model, args.debug, args.dry_mode, args.from_scratch, args.domains)