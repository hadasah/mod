from slurm_jobs.slurm_job import run_grid
from slurm_jobs.slurm_constants import *
import os
import re
import argparse
from slurm_jobs.model_specs import EVAL_FOLDERS
from pathlib import Path

def main(model, load_from_step, domains, data_bin=None, debug=False, dry_mode=False, init_id=None):
    username = os.getlogin()
    if username not in CONSTANTS:
        raise Error("username isn't defined in slurm_constants file")
    RUN_CONSTANTS = CONSTANTS.get(username)
    MOD_FOLDER = RUN_CONSTANTS.get('MOD_FOLDER')
    if data_bin:
        DATA_BIN = data_bin
    else:
        DATA_BIN = RUN_CONSTANTS.get('DATA_BIN')
    JQ_PATH = RUN_CONSTANTS.get('JQ_PATH')

    DEBUG_MODE = debug
    DRY_MODE = dry_mode
    name_keys = []
    NUM_GPUS = 8

    if not domains:
        DOMAINS = [i for i in range(16)]
    else:
        DOMAINS = domains
    # make sure all specified domains exist in data-bin folder
    # assert all([Path(DATA_BIN) / x in Path(DATA_BIN).glob("*/") for x in DOMAINS])

    # Used to distinguish between my naming conventions for demix vs modular models
    MODEL_TYPE = 'mod'
    # Determines where the posteriors and results gets saved

    DOMAIN_PHRASE = f'_INIT={init_id}' if init_id else ''
    EVAL_FOLDER_ID = f'Base_dense_LOAD_FROM_STEP_{load_from_step}_LR_0.0005{DOMAIN_PHRASE}'
    # Comma separated list of the checkpoint IDs. 
    #Unfortunately this can't be set per job, I'm assuming we're always setting the right # updates
    CHECKPOINT_IDS = 'last,last,last,last,last,last,last,last'
    # CHECKPOINT_IDS = 'best,best,best,best,best,best,best,best'

    EVAL_SCRIPT = f'{MOD_FOLDER}/demix/mix_eval_pipeline.sh' if MODEL_TYPE in ['demix', 'mod'] else f'{MOD_FOLDER}/demix/eval_pipeline.sh'
    MODEL=model
    SWEEP_NAME = f"eval_sweep_{MODEL}_mod_LOAD_FROM_STEP_{load_from_step}"
    EVAL_FOLDER = EVAL_FOLDERS[MODEL]
    ROOT_MODEL_FOLDER = EVAL_FOLDER[MODEL_TYPE]
    if MODEL_TYPE == 'mod':
        ROOT_MODEL_FOLDER = ROOT_MODEL_FOLDER.format(DOMAIN_PHRASE)
        print(ROOT_MODEL_FOLDER)

    grids = {
        SWEEP_NAME: {
            'fixed_args': '',
            'positional_args': {
                "NUM_GPUS": [NUM_GPUS],
                "DATA_BIN": [DATA_BIN],
                "ROOT_MODEL_FOLDER": [ROOT_MODEL_FOLDER],
                "MODEL_FOLDER": ['.'],
                "CHECKPOINT_IDS": [CHECKPOINT_IDS],
                "DOMAIN_ID": DOMAINS,
                "ENSEMBLE_TYPE": ['cached_prior'],
                "MODEL_TYPE": [MODEL_TYPE],
                # "GENERALIST_MODEL": ["/checkpoint/suching/margaret_sweep_rerun/small/_EXPERIMENT=dense_NUMSTEPS=36000_LR=0.001/checkpoint_1_30000.pt"],
                "GENERALIST_MODEL": ["None"],
                "TOP_K": [8],
                "EVAL_FOLDER_ID": [EVAL_FOLDER_ID],
                "LOAD_FROM_STEP": [load_from_step],
                "EXCLUDE_EXPERT": ["False"],
                "ONLY_USE_DOMAIN_EXPERT": ['False'],
                "MODEL": [model],
                "MOD_FOLDER": [MOD_FOLDER],
                "JQ_PATH": [JQ_PATH]
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
            prefix=f'bash {EVAL_SCRIPT}',
            gpus=NUM_GPUS,
            cpus=10,
            nodes=1,
            #TODO change these
            account=RUN_CONSTANTS.get('SLURM_ACCOUNT'),
            # partition=RUN_CONSTANTS.get('SLURM_PARTITION'),
            partition="scavenge",
            jobtime='3:00:00',
            mem_gb=480,
            job_id_start=1,
            debug_mode=DEBUG_MODE,
            dry_mode=DRY_MODE,
            DIR_PATH=MOD_FOLDER,
            #TODO change this
            conda_env_name=RUN_CONSTANTS['CONDA_ENV'],
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dry-mode', action='store_true')
    parser.add_argument('--model', type=str)
    parser.add_argument('--domains', type=str, nargs="+")
    parser.add_argument('--load-from-step', type=int)
    parser.add_argument('--data-bin', type=str)
    parser.add_argument('--init-id', type=str)
    args = parser.parse_args()
    main(args.model,  args.load_from_step, args.domains, args.data_bin, args.debug, args.dry_mode, args.init_id)
