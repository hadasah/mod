from slurm_jobs.slurm_job import run_grid
import os
import re
from slurm_jobs.slurm_constants import *
from slurm_jobs.model_specs import EVAL_FOLDERS, DOMAINS
import argparse
from pathlib import Path

def main(model, domains, path_to_model=None, path_to_model_dir=None, checkpoint_id=None, model_regex=None, data_bin=None, debug=False, dry_mode=False, tag=None, eval_tag="", output_dir=None):

    DEBUG_MODE = debug
    DRY_MODE = dry_mode
    name_keys = []
    NUM_GPUS = 1
    NUM_NODES = 1
    username = os.getlogin()
    if username not in CONSTANTS:
            raise Error("username isn't defined in slurm_constants file")
    RUN_CONSTANTS = CONSTANTS.get(username)
    MOD_FOLDER = RUN_CONSTANTS.get('MOD_FOLDER')
    # MODEL_FOLDER = RUN_CONSTANTS.get('MODEL_FOLDER') + "/small/"

    # MODEL_FOLDER = "/checkpoint/suching/mod/_modular_gpt3_small_80K/modular_gpt3_small_80K_LR=0.0005/"

    if path_to_model is not None:
        MODEL_FOLDER = path_to_model
        SWEEP_NAME = f"eval_sweep_average_{eval_tag}"
        selected_folders = "."
    elif path_to_model_dir is not None:
        MODEL_FOLDER = path_to_model_dir
        SWEEP_NAME = f"eval_sweep_average_{eval_tag}"
        all_runs = os.listdir(MODEL_FOLDER)
        if model_regex is not None:
            regex = re.compile(model_regex)
            selected_folders = [folder for folder in all_runs if regex.match(folder)]
            print(selected_folders)
        else:
            selected_folders = "."
    else:
        if not tag:
            tag = 'dense'
        MODEL=model
        SWEEP_NAME = f"eval_sweep_{MODEL}_{tag}_{eval_tag}"
        EVAL_FOLDER = EVAL_FOLDERS[MODEL]
        MODEL_FOLDER = EVAL_FOLDER[tag]
        selected_folders = "."

    if domains[0] in DOMAINS.keys():
        domains = DOMAINS[domains[0]]
    if data_bin:
        DATA_BIN = data_bin
    else:
        DATA_BIN = RUN_CONSTANTS.get('DATA_BIN')
    JQ_PATH = RUN_CONSTANTS.get('JQ_PATH')

    # make sure all specified domains exist in data-bin folder
    if not all([Path(DATA_BIN) / x in Path(DATA_BIN).glob("*/") for x in domains]):
        print([Path(DATA_BIN) / x for x in domains if Path(DATA_BIN) / x not in Path(DATA_BIN).glob("*/")])
        assert False



    # This regex looks in MODEL_FOLDER's subfolders for matches
    # WANTED_FOLDER_REGEX = 'MODEL=transformerlmgpt3small_DOMAINID=0_PHASEONERATIO=0.6_RESETITEMS=dataloader_UPDATEFREQ=32_LR=0.0005'
    # WANTED_FOLDER_REGEX='merged_mod'
    # Used to distinguish between my naming conventions for demix vs modular models
    MODEL_TYPE = 'dense'
    # Determines where the posteriors and results gets saved 
    EVAL_FOLDER_ID = 'Base_dense'
    # Comma separated list of the checkpoint IDs. 
    #Unfortunately this can't be set per job, I'm assuming we're always setting the right # updates
    CHECKPOINT_ID = checkpoint_id or 'last'

    EVAL_SCRIPT = f'{MOD_FOLDER}/demix/mix_eval_pipeline.sh' if MODEL_TYPE in ['demix', 'modular'] else f'{MOD_FOLDER}/demix/eval_pipeline.sh'
    # all_runs = os.listdir(MODEL_FOLDER)
    # regex = re.compile(WANTED_FOLDER_REGEX)
    # selected_folders = [folder for folder in all_runs if regex.match(folder)]

    grids = {
        SWEEP_NAME: {
            'fixed_args': '',
            'positional_args': {
                "DATA_BIN": [DATA_BIN],
                "ROOT_MODEL_FOLDER": [MODEL_FOLDER],
                "MODEL_FOLDERS": selected_folders,
                "CHECKPOINT_ID": [CHECKPOINT_ID],
                "SPLIT": ['test'],
                "DOMAIN_ID": domains,
                "MOD_FOLDER": [MOD_FOLDER],
                "FORCE_DOMAIN_TOKEN": ["FALSE"],
                "EVAL_TAG": [eval_tag],
                "OUTPUT_DIR": [output_dir]
            },
            'named_args': {},
        },
    }
    if path_to_model is not None or path_to_model_dir is not None:
        volta32=False
        mem_gb=40
        jobtime='2:00:00'
    else:
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
            mem_gb=mem_gb,
            volta32=volta32,
            job_id_start=1,
            debug_mode=DEBUG_MODE,
            dry_mode=DRY_MODE,
            DIR_PATH=MOD_FOLDER,
            conda_env_name='mod',
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dry-mode', action='store_true')
    parser.add_argument('--path-to-model', type=str)
    parser.add_argument('--path-to-model-dir', type=str)
    parser.add_argument('--model-regex', type=str, default=None)
    parser.add_argument('--checkpoint-id', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--eval-tag', type=str, default=None)
    parser.add_argument('--domains', type=str, nargs="+")
    parser.add_argument('--data-bin', type=str)
    parser.add_argument('--output-dir', type=str)
    args = parser.parse_args()
    main(args.model,   args.domains, args.path_to_model, args.path_to_model_dir, args.checkpoint_id, args.model_regex, args.data_bin, args.debug, args.dry_mode, args.tag, args.eval_tag,args.output_dir)

