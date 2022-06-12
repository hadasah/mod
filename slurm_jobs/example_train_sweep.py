import argparse
import numpy as np
import os

from slurm_jobs.model_specs import *
from slurm_jobs.slurm_constants import *
from slurm_jobs.slurm_job import run_grid


def main(model, experiment, debug=False, dry_mode=False, domains=None):
    DEBUG_MODE = debug
    DRY_MODE = dry_mode
    MODEL = model

    username = os.getlogin()
    RUN_CONSTANTS = CONSTANTS.get(username)
    if RUN_CONSTANTS is None:
        raise Error("username isn't defined in slurm_constants file")
    MOD_FOLDER = RUN_CONSTANTS.get('MOD_FOLDER')
    SPECS = SPECS[MODEL]
    NUM_GPUS = SPECS['NUM_GPUS']
    NUM_NODES = SPECS['NUM_GPUS'] // 8
    SWEEP_NAME = f"sweep_{MODEL}_{SPECS['NUM_GPUS']}_GPUs"
    name_keys = ["EXPERIMENT", "NUM_STEPS", "UPDATE_FREQ", "LR", "NUM_GPUS"]
    
    if domains is None:
        domains = ["0,1,2,3,4,5,6,7,8"]
        # domains = secondary_domains
    else:
        if domains == 'secondary':
            domains = secondary_domains
        name_keys.append("DOMAIN_IDS")
    
    if domains[0] in VALID:
        DATA_BIN = RUN_CONSTANTS.get('DATA_BIN')
    else:
        DATA_BIN = RUN_CONSTANTS.get("SECONDARY_DATA_BIN")

    grids = {
        SWEEP_NAME: {
            'fixed_args': '',
            'positional_args': {
                "SWEEP_NAME": [SWEEP_NAME],
                "NUM_GPUS": [SPECS['NUM_GPUS']],
                "MODEL": ['transformer_lm_gpt3_small'],
                "EXPERIMENT": ['demix'],
                "DATA_PATH": [RUN_CONSTANTS.get('DATA_BIN')],
                "DOMAIN_IDS": domains,
                "PARAMS_TO_FREEZE": ["None"],
                "COPYING_MODEL_FOLDER": ["None"],
                "MODEL_FOLDER": [RUN_CONSTANTS.get('MODEL_FOLDER')],
                "SUBFOLDER_NAME": ["None"],
                "STOP_TIME_HOURS": [SPECS['TRAIN_HOURS']],
                "PHASE_ONE_RATIO": ["None"],
                "PHASE_ONE_UPDATE_NUM": ["None"],
                "RESET_ITEMS": ["None"],
                "NUM_STEPS": [SPECS['NUM_STEPS']],
                "UPDATE_FREQ": [SPECS["UF"]],
                "LR": [SPECS["LR"]],
                "SAVE_INTERVAL_UPDATES": [SPECS['SAVE_INTERVAL_UPDATES']],
                "DISTRIBUTED_PORT": [np.random.randint(1024, 65535)],
                "WANDB_PROJECT": ['publication-test'],
                "WANDB_ENTITY": ['scaling-demix'],
                "MOD_FOLDER": [MOD_FOLDER],
        },
    }

    for sweep_name, grid in grids.items():
        run_grid(
            grid,
            name_keys,
            sweep_name,
            user=os.environ['USER'],
            prefix=f'bash {MOD_FOLDER}/demix/train.sh',
            gpus=NUM_GPUS,
            cpus=RUN_CONSTANTS.get('NUM_CPUS'),
            nodes=NUM_NODES,
            #TODO change these
            account=RUN_CONSTANTS['SLURM_ACCOUNT'],
            partition=RUN_CONSTANTS['SLURM_PARTITION'],
            jobtime='72:00:00',
            mem_gb=480,
            job_id_start=1,
            debug_mode=DEBUG_MODE,
            dry_mode=DRY_MODE,
            add_name='end',
            DIR_PATH=MOD_FOLDER,
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dry-mode', action='store_true')
    parser.add_argument('--model', type=str)
    parser.add_argument('--experiment', type=str)
    parser.add_argument('--domains', type=str, nargs="+")
    args = parser.parse_args()
    main(args.model, args.experiment, args.debug, args.dry_mode, args.domains)
