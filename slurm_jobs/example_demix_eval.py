from slurm_jobs.slurm_job import run_grid
from slurm_jobs.slurm_constants import *
import argparse
import os
import re

SWEEP_NAME = "eval_sweep_gpt3_small_demix"
DEBUG_MODE = False
DRY_MODE = False
name_keys = []
NUM_GPUS = 8
NUM_NODES = 1


def main(model, load_from_step, domains, data_bin=None, debug=False, dry_mode=False):
    username = os.getlogin()
    if username not in CONSTANTS:
            raise Error("username isn't defined in slurm_constants file")
    RUN_CONSTANTS = CONSTANTS.get(username)
    MOD_FOLDER = RUN_CONSTANTS.get('MOD_FOLDER')
    #MODEL_FOLDER = RUN_CONSTANTS.get('MODEL_FOLDER') + "/small/"
    DATA_BIN = RUN_CONSTANTS.get('DATA_BIN')
    JQ_PATH = RUN_CONSTANTS.get('JQ_PATH')
    MODEL_FOLDER = "/checkpoint/margaretli/mod_publication/"


    # This regex looks in MODEL_FOLDER's subfolders for matches
    WANTED_FOLDER_REGEX = '.*EXPERIMENT=demix.*'
    # Used to distinguish between my naming conventions for demix vs modular models
    MODEL_TYPE = 'demix'
    # Determines where the posteriors and results gets saved 
    EVAL_FOLDER_ID = 'Base_med_demix'
    # Comma separated list of the checkpoint IDs. 
    #Unfortunately this can't be set per job, I'm assuming we're always setting the right # updates
    CHECKPOINT_IDS = 'best,best,best,best,best,best,best,best'

    EVAL_SCRIPT = f'{MOD_FOLDER}/demix/mix_eval_pipeline.sh' if MODEL_TYPE in ['demix', 'modular'] else f'{MOD_FOLDER}/demix/eval_pipeline.sh'
    all_runs = os.listdir(MODEL_FOLDER)
    regex = re.compile(WANTED_FOLDER_REGEX)
    selected_folders = [folder for folder in all_runs if regex.match(folder)]

    if not domains:
        DOMAINS = [i for i in range(16)]
    else:
        DOMAINS = domains

    print(selected_folders)
    grids = {
        SWEEP_NAME: {
            'fixed_args': '',
            'positional_args': {
                "NUM_GPUS": [NUM_GPUS],
                "DATA_BIN": [DATA_BIN],
                "ROOT_MODEL_FOLDER": [MODEL_FOLDER],
                "MODEL_FOLDERS": selected_folders,
                "CHECKPOINT_IDS": [CHECKPOINT_IDS],
                "DOMAIN_ID": DOMAINS,
                "ENSEMBLE_TYPE": ['cached_prior'],
                "MODEL_TYPE": [MODEL_TYPE],
                "GENERALIST_MODEL": ["None"],
                "TOP_K": [8],
                "EVAL_FOLDER_ID": [EVAL_FOLDER_ID],
                "NUM_STEPS": ["None"],
                "EXCLUDE_EXPERT": ["False"],
                "ONLY_USE_DOMAIN_EXPERT": ['False'],
                "ARCH": ['transformer_lm_gpt3_medium'],
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
            nodes=NUM_NODES,
            #TODO change these
            account=RUN_CONSTANTS['SLURM_ACCOUNT'],
            partition=RUN_CONSTANTS['SLURM_PARTITION'],
            jobtime='2:00:00',
            mem_gb=40,
            job_id_start=1,
            debug_mode=DEBUG_MODE,
            dry_mode=DRY_MODE,
            DIR_PATH=MOD_FOLDER,
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
    args = parser.parse_args()
    main(args.model,  args.load_from_step, args.domains, args.data_bin, args.debug, args.dry_mode)
