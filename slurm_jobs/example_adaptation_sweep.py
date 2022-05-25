from mod_utils import mod_checkpoint_utils
from slurm_jobs.slurm_constants import CONSTANTS
from slurm_jobs.slurm_job import run_grid
import fairseq
import os
import argparse
import numpy as np
from pathlib import Path


def main(model, debug=False, dry_mode=False, from_scratch=False, domains=None, data_bin=None, load_from_step=-1, average=False, average_weights=None):
    DEBUG_MODE = debug
    DRY_MODE = dry_mode
    FROM_SCRATCH = from_scratch
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
    
    SWEEP_NAME = f"sweep_{MODEL}_adaptation_average_{average}"


    name_keys = ["MODEL",  "LOAD_FROM_STEP", "RESET_ITEMS", "LR", "UPDATE_FREQ", "DOMAIN_ID", "AVERAGE"]
    from slurm_jobs.model_specs import SPECS
    SPECS = SPECS[MODEL]
    NUM_GPUS = SPECS['NUM_MOD_GPUS']

    NUM_NODES = 1 if NUM_GPUS < 8 else NUM_GPUS // 8

    # modify to path to dense checkpoint you want to use
    PATH_TO_DENSE_CHECKPOINTS = f'/checkpoint/suching/mod/_modular_gpt3_small_80K/modular_gpt3_small_80K_LR\=0.0005/MODEL=transformerlmgpt3small_DOMAINID=1_LOADFROMSTEP=24000_RESETITEMS=dataloader_UPDATEFREQ=32_LR=0.0005/'
    NEW_MODEL_TOP_FOLDER = f'/checkpoint/suching/mod/_adaptation_{MODEL}/adaptation_{MODEL}_LR={SPECS["LR"]}/'

    re_string = ''
    if not domains:
        DOMAINS = [i for i in range(8)]
    else:
        DOMAINS = domains

    if average_weights:
        average_weights = ",".join(map(str, average_weights))
    else:
        average_weights = None
    if average:
        average = "True"
    else:
        average = "False"
    grids = {
        SWEEP_NAME: {
            'fixed_args': '',
            'positional_args': {
                "NUM_GPUS": [NUM_GPUS],
                "MODEL": [MODEL],
                "DATA_BIN": [DATA_BIN],
                "DOMAIN_ID": DOMAINS,
                "PARAMS_TO_FREEZE": ["None"],
                "COPYING_MODEL_FOLDER": [PATH_TO_DENSE_CHECKPOINTS],
                "NEW_MODEL_TOP_FOLDER": [NEW_MODEL_TOP_FOLDER],
                "CHECKPOINTS_SUBFOLDER": '.',
                "LOAD_FROM_STEP": [-1],
                "RESET_ITEMS": ['meters,dataloader,optimizer,lr-scheduler'],
                "NUM_STEPS": [10000],
                "AVERAGE": [average],
                "AVERAGE_WEIGHTS": [average_weights],
                "STOP_TIME_HOURS": [60],
                "UPDATE_FREQ": [SPECS['UF']],

                "LR": [SPECS['LR'] * .1],
                "PORT": [np.random.randint(1024, 65535)],
                "WANDB_PROJECT": ['mod_test'],
                "WANDB_ENTITY": ['scaling-demix'],
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
            prefix=f'bash {MOD_FOLDER}/demix/adaptation.sh',
            gpus=NUM_GPUS,
            cpus=RUN_CONSTANTS.get('NUM_CPUS'),
            nodes=NUM_NODES,
            account=RUN_CONSTANTS.get('SLURM_ACCOUNT'),
            partition=RUN_CONSTANTS.get('SLURM_PARTITION'),
            jobtime="50:00:00",
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
    parser.add_argument('--domains', type=str, nargs="+")
    parser.add_argument('--load-from-step', type=int, default=-1)
    parser.add_argument('--from-scratch', action='store_true')
    parser.add_argument('--average', action='store_true')
    parser.add_argument('--average-weights', type=float, nargs="+")
    parser.add_argument('--data-bin', type=str)

    args = parser.parse_args()
    main(args.model, args.debug, args.dry_mode, args.from_scratch, args.domains, args.data_bin, args.load_from_step, args.average, args.average_weights)
