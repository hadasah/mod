from slurm_jobs.slurm_job import run_grid
import argparse
from slurm_jobs.slurm_constants import CONSTANTS
import os
import numpy as np
from pathlib import Path


def main(model, experiment, domains, data_bin, debug=False, dry_mode=False, id_=''):
    DEBUG_MODE = debug
    DRY_MODE = dry_mode
    MODEL = model
    
    
    username = os.getlogin()
    RUN_CONSTANTS = CONSTANTS.get(username)
    if RUN_CONSTANTS is None:
        raise Error("username isn't defined in slurm_constants file")
    MOD_FOLDER = RUN_CONSTANTS.get('MOD_FOLDER')
    from slurm_jobs.model_specs import SPECS
    SPECS = SPECS[MODEL]
    if data_bin:
        DATA_BIN = data_bin
    else:
        DATA_BIN = RUN_CONSTANTS.get('DATA_BIN')

    NUM_NODES = SPECS['NUM_GPUS'] // 8
    SWEEP_NAME = f"sweep_{MODEL}_{SPECS['NUM_GPUS']}_GPUs"
    
    name_keys = ["EXPERIMENT", "MODEL", "STOP_TIME_HOURS", "NUM_STEPS", "UPDATE_FREQ", "LR", "NUM_GPUS", "ID"]
    
    if not all([Path(DATA_BIN) / x in Path(DATA_BIN).glob("*/") for x in domains]):
        print([Path(DATA_BIN) / x for x in domains if Path(DATA_BIN) / x not in Path(DATA_BIN).glob("*/")])
        assert False


    grids = {
        SWEEP_NAME: {
            'fixed_args': '',
            'positional_args': {
                "NUM_GPUS": [SPECS['NUM_GPUS']],
                "DISTRIBUTED_PORT": [np.random.randint(1024, 65535)],
                "MODEL": [MODEL],
                "EXPERIMENT": [experiment],
                "MODEL_DIR": [RUN_CONSTANTS.get('MODEL_FOLDER')],
                "DATA_BIN": [DATA_BIN],
                "NUM_STEPS": [SPECS['NUM_STEPS']],
                "SAVE_INTERVAL_UPDATES": [SPECS['SAVE_INTERVAL_UPDATES']],
                "DOMAINS": [",".join(domains)],
                "VALID_SUBSET": [','.join(["valid_" + x for x in domains])],
                "STOP_TIME_HOURS": [SPECS['TRAIN_HOURS']],
                "UPDATE_FREQ": [SPECS["UF"]],
                "LR": [SPECS["LR"]],
                "WANDB_PROJECT": ['publication-test'],
                "WANDB_ENTITY": ['scaling-demix'],
                "MOD_FOLDER": [MOD_FOLDER],
                "ID": [id_]
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
            prefix=f'bash {MOD_FOLDER}/demix/train.sh',
            gpus=8,
            cpus=10,
            nodes=NUM_NODES,
            #TODO change these
            account=RUN_CONSTANTS['SLURM_ACCOUNT'],
            partition=RUN_CONSTANTS['SLURM_PARTITION'],
            jobtime='72:00:00',
            mem_gb=480,
            job_id_start=1,
            volta32=True,
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
    parser.add_argument('--id', type=str)
    parser.add_argument('--domains', type=str, nargs="+")
    parser.add_argument('--data-bin', type=str, default='/private/home/suching/raw_data/data-bin-big/')

    parser.add_argument('--experiment', type=str)
    args = parser.parse_args()
    main(args.model, args.experiment, args.domains, args.data_bin, args.debug, args.dry_mode, args.id)
