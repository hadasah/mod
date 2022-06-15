from slurm_jobs.slurm_job import run_grid
from slurm_jobs.slurm_constants import *
import os
import re
import numpy as np
import argparse
from slurm_jobs.model_specs import EVAL_FOLDERS
from pathlib import Path

def main(model, load_from_step, evaluation_domains, additional_domains=None, data_bin=None, debug=False, dry_mode=False, dev_posteriors_dir=None):
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

    SWEEP_NAME = f"eval_sweep_{model}_average_mod_LOAD_FROM_STEP_{load_from_step}"
    DEBUG_MODE = debug
    DRY_MODE = dry_mode
    name_keys = []
    NUM_GPUS = 1

    # make sure all specified domains exist in data-bin folder
    if not all([Path(DATA_BIN) / x in Path(DATA_BIN).glob("*/") for x in evaluation_domains]):
        print([Path(DATA_BIN) / x for x in evaluation_domains if Path(DATA_BIN) / x not in Path(DATA_BIN).glob("*/")])
        assert False
    # /checkpoint/suching/suchin_mod_8_GPUs/small/_EXPERIMENT=dense_NUMSTEPS=36000_LR=0.001/_DOMAIN_5_MOD_STEPS_30000_PHASE1_DENSE
    # /checkpoint/suching/mod_sweep/_modular_gpt3_small_36K/modular_gpt3_small_36K_LR=0.001/MODEL=transformerlmgpt3small_DOMAINID=7_PHASEONERATIO=0.25_RESETITEMS=dataloader,meters_UPDATEFREQ=32_LR=0.001

    # MODEL=transformerlmgpt3small_DOMAINID=7_PHASEONERATIO=0.25_RESETITEMS=dataloader,meters_UPDATEFREQ=32_LR=0.001

    # This regex looks in MODEL_FOLDER's subfolders for matches
    WANTED_FOLDER_REGEX = '.*'
    # Used to distinguish between my naming conventions for demix vs modular models
    MODEL_TYPE = 'modular'
    # Determines where the posteriors and results gets saved 
    EVAL_FOLDER_ID = f'Base_average_mod_LR_0.0005'
    # Comma separated list of the checkpoint IDs. 
    #Unfortunately this can't be set per job, I'm assuming we're always setting the right # updates
    CHECKPOINT_IDS = ",".join(['last'] * (len(additional_domains.split(',')) + 8))
    EVAL_SCRIPT = f'{MOD_FOLDER}/demix/average_eval_pipeline.sh' if MODEL_TYPE in ['demix', 'modular'] else f'{MOD_FOLDER}/demix/eval_pipeline.sh'
    # all_runs = os.listdir("/checkpoint/suching/mod/_modular_transformer_lm_gpt3_medium/modular_transformer_lm_gpt3_medium_LR=0.0005/")
    # regex = re.compile(WANTED_FOLDER_REGEX)
    # selected_folders = [folder for folder in all_runs if regex.match(folder)]
    # print(selected_folders)

    MODEL=model
    SWEEP_NAME = f"eval_sweep_{MODEL}_average_mod"
    EVAL_FOLDER = EVAL_FOLDERS[MODEL]
    ROOT_MODEL_FOLDER = EVAL_FOLDER['mod']

    # if model == 'transformer_lm_gpt3_small':
    #     ROOT_MODEL_FOLDER = "/checkpoint/suching/mod/_modular_gpt3_small_80K/modular_gpt3_small_80K_LR=0.0005/"
    # elif model == 'transformer_lm_gpt3_medium':
    #     ROOT_MODEL_FOLDER = "/checkpoint/suching/mod/_modular_transformer_lm_gpt3_medium/modular_transformer_lm_gpt3_medium_LR=0.0005/"
    if not dev_posteriors_dir:
        dev_posteriors_dir = f"{ROOT_MODEL_FOLDER}/evals_top8_Base_dense_LOAD_FROM_STEP_${load_from_step}_LR_0.0005/"

    grids = {
        SWEEP_NAME: {
            'fixed_args': '',
            'positional_args': {
                "NUM_GPUS": [NUM_GPUS],
                "DATA_BIN": [DATA_BIN],
                "ROOT_MODEL_FOLDER": [ROOT_MODEL_FOLDER],
                "MODEL_FOLDER": ['.'],
                "CHECKPOINT_IDS": [CHECKPOINT_IDS],
                "DOMAIN_ID": evaluation_domains,
                "ENSEMBLE_TYPE": ['cached_prior'],
                "MODEL_TYPE": [MODEL_TYPE],
                # "GENERALIST_MODEL": ["/checkpoint/suching/margaret_sweep_rerun/small/_EXPERIMENT=dense_NUMSTEPS=36000_LR=0.001/checkpoint_1_30000.pt"],
                "GENERALIST_MODEL": ["None"],
                "TOP_K": [8],
                "EVAL_FOLDER_ID": [EVAL_FOLDER_ID],
                "LOAD_FROM_STEP": load_from_step,
                "EXCLUDE_EXPERT": ["False"],
                "ONLY_USE_DOMAIN_EXPERT": ['False'],
                "MODEL": [model],
                "OUTPUT_DIR": [f"/checkpoint/suching/mod/average_{model}/"],
                "MOD_FOLDER": [MOD_FOLDER],
                "JQ_PATH": [JQ_PATH],
                "RANDOM_JOB_PORT": [np.random.randint(5,100)],
                "DEV_POSTERIORS_DIR": [dev_posteriors_dir],
                "ADDITIONAL_DOMAINS": [additional_domains]
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
            partition=RUN_CONSTANTS.get('SLURM_PARTITION'),
            jobtime='2:00:00',
            volta32=True,
            mem_gb=480,
            job_id_start=1,
            debug_mode=DEBUG_MODE,
            dry_mode=DRY_MODE,
            DIR_PATH=MOD_FOLDER,
            #TODO change this
            conda_env_name='latest',
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dry-mode', action='store_true')
    parser.add_argument('--model', type=str)
    parser.add_argument('--evaluation-domains', type=str, nargs="+")
    parser.add_argument('--additional-domains', type=str, nargs="+")
    parser.add_argument('--load-from-step', type=int, nargs="+")
    parser.add_argument('--data-bin', type=str)
    parser.add_argument('--dev-posteriors-dir', type=str)
    args = parser.parse_args()
    main(args.model,  args.load_from_step, args.evaluation_domains, ",".join(args.additional_domains), args.data_bin, args.debug, args.dry_mode, args.dev_posteriors_dir)