from mod_utils import mod_checkpoint_utils
from slurm_jobs.slurm_constants import CONSTANTS
from slurm_jobs.model_specs import DOMAINS
from slurm_jobs.slurm_job import run_grid
import fairseq
import os
import argparse
import numpy as np
from pathlib import Path
import re


def main(model, domains, debug=False, dry_mode=False, from_scratch=False, data_bin=None, load_from_step=-1, average=False, average_weights=None, path_to_averages=None):
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

    if domains[0] in DOMAINS.keys():
        domains = DOMAINS[domains[0]]
    if not all([Path(DATA_BIN) / x in Path(DATA_BIN).glob("*/") for x in domains]):
        print([Path(DATA_BIN) / x for x in domains if Path(DATA_BIN) / x not in Path(DATA_BIN).glob("*/") ])
        assert False
    
    SWEEP_NAME = f"sweep_{MODEL}_adaptation_average_{average}"


    name_keys = ["MODEL",  "LOAD_FROM_STEP", "RESET_ITEMS", "LR", "UPDATE_FREQ", "DOMAIN_ID", "AVERAGE", "NUM_STEPS"]
    from slurm_jobs.model_specs import SPECS, EVAL_FOLDERS
    SPECS = SPECS[MODEL]
    NUM_GPUS = SPECS['NUM_MOD_GPUS']

    NUM_NODES = 1 if NUM_GPUS < 8 else NUM_GPUS // 8

    # modify to path to dense checkpoint you want to use
    if not path_to_averages:
        path_to_averages = EVAL_FOLDERS[MODEL]["average"]
    PATH_TO_DENSE_CHECKPOINTS = Path(path_to_averages) / f"_LOAD_FROM_STEP_{load_from_step}/"
    NEW_MODEL_TOP_FOLDER = f'/checkpoint/suching/mod/_adaptation_{MODEL}/adaptation_{MODEL}_LR={SPECS["LR"]}/'

    # re_string = ''
    # WANTED_FOLDER_REGEX=f"DOMAINID\={domain}"
    # regex = re.compile(WANTED_FOLDER_REGEX)
    # all_runs = os.listdir(PATH_TO_DENSE_CHECKPOINTS)
    # print(all_runs)
    # assert False
    # selected_folders = [folder for folder in all_runs if regex.match(folder)]
    # print(selected_folders)
    
    grids = {
        SWEEP_NAME: {
            'fixed_args': '',
            'positional_args': {
                "NUM_GPUS": [NUM_GPUS],
                "MODEL": [MODEL],
                "DATA_BIN": [DATA_BIN],
                "DOMAIN_ID": domains,
                "PARAMS_TO_FREEZE": ["None"],
                "COPYING_MODEL_FOLDER": [str(PATH_TO_DENSE_CHECKPOINTS)],
                "NEW_MODEL_TOP_FOLDER": [NEW_MODEL_TOP_FOLDER],
                "CHECKPOINTS_SUBFOLDER": ".",
                "LOAD_FROM_STEP": [-1],
                "RESET_ITEMS": ['optimizer,meters,lr-scheduler,dataloader'],
                "NUM_STEPS": [16000],
                "AVERAGE": [average],
                "AVERAGE_WEIGHTS": [average_weights],
                "STOP_TIME_HOURS": [SPECS["TRAIN_HOURS"] + SPECS["TRAIN_HOURS"] // 2],
                "UPDATE_FREQ": [SPECS['UF']],
                "LR": [SPECS['LR'] * 0.1],
                "RANDOM_JOB_PORT": [np.random.randint(5,100)],
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
    parser.add_argument('--domains', type=str, nargs='+')
    # parser.add_argument('--target-domain', type=str)
    parser.add_argument('--load-from-step', type=int, default=-1)
    parser.add_argument('--from-scratch', action='store_true')
    parser.add_argument('--data-bin', type=str)
    parser.add_argument('--path-to-averages', type=str)

    args = parser.parse_args()
    main(args.model, args.domains, args.debug, args.dry_mode, args.from_scratch, args.data_bin, args.load_from_step, path_to_averages=args.path_to_averages)
