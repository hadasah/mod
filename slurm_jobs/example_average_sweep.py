from mod_utils import mod_checkpoint_utils
from slurm_jobs.slurm_constants import CONSTANTS
from slurm_jobs.slurm_job import run_grid
import fairseq
import os
import argparse
import numpy as np
from pathlib import Path


def main(model, debug=False, dry_mode=False, domains=None, data_bin=None):
    DEBUG_MODE = debug
    DRY_MODE = dry_mode

    MODEL = model
    username = os.getlogin()
    if username not in CONSTANTS:
        raise Error("username isn't defined in slurm_constants file")
    RUN_CONSTANTS = CONSTANTS.get(username)
    MOD_FOLDER = RUN_CONSTANTS.get('MOD_FOLDER')
    if data_bin:
        DATA_BIN = data_bin
    else:
        DATA_BIN = RUN_CONSTANTS.get('DATA_BIN')
    assert all([Path(DATA_BIN) / x in Path(DATA_BIN).glob("*/") for x in domains])
    
    SWEEP_NAME = f"sweep_{MODEL}_average"


    name_keys = ["MODEL",  "LOAD_FROM_STEP", "DOMAIN_ID", "AVERAGE", "TOPK", "UNIFORM"]
    from slurm_jobs.model_specs import SPECS
    SPECS = SPECS[MODEL]


    
    re_string = ''
    if not domains:
        DOMAINS = [i for i in range(16)]
    else:
        DOMAINS = domains

    NEW_MODEL_TOP_FOLDER = f'/checkpoint/suching/mod/model_averaging/average_{MODEL}/'
    grids = {
        # SWEEP_NAME + "_3": {
        #     'fixed_args': '',
        #     'positional_args': {
        #         "NUM_GPUS": [8],
        #         "MODEL": [MODEL],
        #         "DATA_BIN": [DATA_BIN],
        #         "DOMAIN_ID": DOMAINS,
        #         "COPYING_MODEL_FOLDER": [PATH_TO_DENSE_CHECKPOINTS],
        #         "NEW_MODEL_TOP_FOLDER": [NEW_MODEL_TOP_FOLDER],
        #         "CHECKPOINTS_SUBFOLDER": '.',
        #         "LOAD_FROM_STEP": [1000, 8000,16000,56000,72000],
        #         "AVERAGE": ["False"],
        #         "TOPK": [8],
        #         "UNIFORM": ["False"],
        #         "PORT": [np.random.randint(1024, 65535)],
        #         "MOD_FOLDER": [MOD_FOLDER]
        #     },
        #     'named_args': {},
        # },
        SWEEP_NAME: {
            'fixed_args': '',
            'positional_args': {
                "NUM_GPUS": [8],
                "MODEL": [MODEL],
                "DATA_BIN": [DATA_BIN],
                "DOMAIN_ID": DOMAINS,
                "NEW_MODEL_TOP_FOLDER": [NEW_MODEL_TOP_FOLDER],
                "CHECKPOINTS_SUBFOLDER": '.',
                "LOAD_FROM_STEP": [15000],
                "AVERAGE": ["True"],
                "TOPK": [8],
                "UNIFORM": ["False"],
                "PORT": [np.random.randint(1024, 65535)],
                "MOD_FOLDER": [MOD_FOLDER]
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
            prefix=f'bash {MOD_FOLDER}/demix/average.sh',
            gpus=grid['positional_args']['NUM_GPUS'][0],
            cpus=RUN_CONSTANTS.get('NUM_CPUS'),
            nodes=1 if grid['positional_args']['NUM_GPUS'][0] < 8 else grid['positional_args']['NUM_GPUS'][0] // 8,
            account=RUN_CONSTANTS.get('SLURM_ACCOUNT'),
            partition=RUN_CONSTANTS.get('SLURM_PARTITION'),
            jobtime="00:10:00",
            mem_gb=100,
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
    parser.add_argument('--domains', type=str, nargs="+")
    parser.add_argument('--data-bin', type=str)

    args = parser.parse_args()
    main(args.model, args.debug, args.dry_mode, args.domains, args.data_bin)
