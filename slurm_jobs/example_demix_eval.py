from slurm_jobs.slurm_job import run_grid
from slurm_jobs.slurm_constants import *
from slurm_jobs.model_specs import EVAL_FOLDERS, DOMAINS
import os
import re
import argparse
from pathlib import Path

def main(domains, model=None, path_to_model_dir=None,  data_bin=None, debug=False, dry_mode=False, tag=None, output_dir=None, checkpoint_id='best'):

    DEBUG_MODE = debug
    DRY_MODE = dry_mode
    name_keys = []
    NUM_GPUS = 8
    NUM_NODES = 1
    username = os.getlogin()
    if username not in CONSTANTS:
            raise Error("username isn't defined in slurm_constants file")
    RUN_CONSTANTS = CONSTANTS.get(username)
    MOD_FOLDER = RUN_CONSTANTS.get('MOD_FOLDER')
    DATA_BIN = data_bin
    JQ_PATH = RUN_CONSTANTS.get('JQ_PATH')
    if domains[0] in DOMAINS.keys():
        domains = DOMAINS[domains[0]]

    if path_to_model_dir is not None:
        SWEEP_NAME = f"eval_sweep_demix"
        MODEL_FOLDER = path_to_model_dir
    else:
        SWEEP_NAME = f"eval_sweep_{MODEL}_{tag}"

        MODEL=model
        if not tag:
            tag = 'demix'
        EVAL_FOLDER = EVAL_FOLDERS[MODEL]
        MODEL_FOLDER = EVAL_FOLDER[tag]
        

# SWEEP_NAME = "eval_sweep_gpt3_small_demix"
# DEBUG_MODE = False
# DRY_MODE = False
# name_keys = []
# NUM_GPUS = 8
# NUM_NODES = 1

# username = os.getlogin()
# if username not in CONSTANTS:
#         raise Error("username isn't defined in slurm_constants file")
# RUN_CONSTANTS = CONSTANTS.get(username)
# MOD_FOLDER = RUN_CONSTANTS.get('MOD_FOLDER')
# #MODEL_FOLDER = RUN_CONSTANTS.get('MODEL_FOLDER') + "/small/"
# DATA_BIN = RUN_CONSTANTS.get('DATA_BIN')
# JQ_PATH = RUN_CONSTANTS.get('JQ_PATH')
# MODEL_FOLDER = "/checkpoint/suching/mod_publication/"


    # This regex looks in MODEL_FOLDER's subfolders for matches
    WANTED_FOLDER_REGEX = '.*demix.*'
    # Used to distinguish between my naming conventions for demix vs modular models
    MODEL_TYPE = 'demix'
    # Determines where the posteriors and results gets saved 
    EVAL_FOLDER_ID = 'Base_demix'
    # Comma separated list of the checkpoint IDs. 
    #Unfortunately this can't be set per job, I'm assuming we're always setting the right # updates
    # CHECKPOINT_IDS =  ",".join([checkpoint_id] * 8)

    EVAL_SCRIPT = f'{MOD_FOLDER}/demix/mix_eval_pipeline.sh' if MODEL_TYPE in ['demix', 'modular'] else f'{MOD_FOLDER}/demix/eval_pipeline.sh'
    # all_runs = os.listdir(MODEL_FOLDER)
    # regex = re.compile(WANTED_FOLDER_REGEX)
    # selected_folders = [folder for folder in all_runs if regex.match(folder)]


    grids = {
        SWEEP_NAME: {
            'fixed_args': '',
            'positional_args': {
                "NUM_GPUS": [NUM_GPUS],
                "DATA_BIN": [DATA_BIN],
                "ROOT_MODEL_FOLDER": [MODEL_FOLDER],
                "MODEL_FOLDERS": '.',
                "CHECKPOINT_IDS": [checkpoint_id],
                "DOMAIN_ID": domains,
                "ENSEMBLE_TYPE": ['cached_prior'],
                "MODEL_TYPE": [MODEL_TYPE],
                "GENERALIST_MODEL": ["None"],
                "TOP_K": [8],
                "EVAL_FOLDER_ID": [EVAL_FOLDER_ID],
                "NUM_STEPS": ["None"],
                "EXCLUDE_EXPERT": ["False"],
                "ONLY_USE_DOMAIN_EXPERT": ['False'],
                "MODEL": [model],
                "OUTPUT_DIR": [output_dir],
                "MOD_FOLDER": [MOD_FOLDER],
                "JQ_PATH": [JQ_PATH],
                "ESTIMATE_POSTERIOR_ONLY": ['False']
            },
            'named_args': {},
        },
    }



    if "xl" in model or "large" in model:
        volta32=True
        mem_gb=140
        jobtime='4:00:00'
    else:
        volta32=False
        mem_gb=40
        jobtime='2:00:00'
    
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
            jobtime=jobtime,
            mem_gb=200,
            job_id_start=1,
            debug_mode=DEBUG_MODE,
            dry_mode=DRY_MODE,
            DIR_PATH=MOD_FOLDER,
            volta32=True,
            #TODO change this
            conda_env_name='mod',
        )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dry-mode', action='store_true')
    parser.add_argument('--model', type=str)
    parser.add_argument('--path-to-model-dir', type=str)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--domains', type=str, nargs="+")
    parser.add_argument('--data-bin', type=str)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--checkpoint-id', type=str)

    args = parser.parse_args()
    main(args.domains, args.model, args.path_to_model_dir,  args.data_bin, args.debug, args.dry_mode, args.tag, args.output_dir, args.checkpoint_id)
