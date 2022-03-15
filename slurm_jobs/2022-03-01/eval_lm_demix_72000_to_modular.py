from slurm_jobs.slurm_job import run_grid
import os

SWEEP_NAME = "eval_demix_72000_to_modular_fixed_update_top3"
DEBUG_MODE = False
DRY_MODE = False
name_keys = []
NUM_GPUS = 3

DATA_BIN='/gscratch/zlab/margsli/gitfiles/demix-data/data-bin'

ROOT_MODEL_FOLDER = '/gscratch/zlab/margsli/demix-checkpoints/models/_demix_72000_to_modular'
MODEL_FOLDER = 'demix_72000_to_modular_DOMAINID='
CHECKPOINT_IDS = '2_66000,1_66000,1_66000,1_66000,1_66000,1_66000,1_66000,1_60000'

grids = {
    SWEEP_NAME: {
        'fixed_args': '',
        'positional_args': {
            "NUM_GPUS": [NUM_GPUS],
            "DATA_BIN": [DATA_BIN],
            "ROOT_MODEL_FOLDER": [ROOT_MODEL_FOLDER],
            "MODEL_FOLDER": [MODEL_FOLDER],
            "CHECKPOINT_IDS": [CHECKPOINT_IDS],
            "DOMAIN_ID": [i for i in range(16)],
            "ENSEMBLE_TYPE": ['cached_prior'],
            "MODEL_TYPE": ["modular"],
            "GENERALIST_MODEL": ["None"],
            "TOP_K": [3],
            "ID": ["fixed_update"],
            "EXCLUDE_DOMAIN_EXPERTS": ["False"],
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
        prefix='bash /mmfs1/gscratch/zlab/margsli/gitfiles/demix/demix/mix_eval_pipeline.sh',
        gpus=NUM_GPUS,
        cpus=5,
        nodes=1,
        account='zlab',
        partition='gpu-rtx6k',
        jobtime='2:00:00',
        mem_gb=40,
        job_id_start=1,
        debug_mode=DEBUG_MODE,
        dry_mode=DRY_MODE,
        # add_name="end",
    )
