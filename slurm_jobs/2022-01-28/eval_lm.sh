from slurm_jobs.slurm_job import run_grid
import os

SWEEP_NAME = "eval_test"
DEBUG_MODE = False
name_keys = []
NUM_GPUS = 8

DATA_BIN='/gscratch/zlab/margsli/gitfiles/demix-data/data-bin'

ROOT_MODEL_FOLDER = '/gscratch/zlab/margsli/gitfiles/demix/models'
MODEL_FOLDER = 'demix_gpt3_small_updatefreq64/demix_8_GPUs_transformer_lm_gpt3_small_updatefreq64'
CHECKPOINT_ID = 'best'

grids = {
    SWEEP_NAME: {
        'fixed_args': '',
        'positional_args': {
            "NUM_GPUS": [8],
            "DATA_BIN": [DATA_BIN],
            "ROOT_MODEL_FOLDER": [ROOT_MODEL_FOLDER],
            "MODEL_FOLDER": [MODEL_FOLDER],
            "CHECKPOINT_ID": [CHECKPOINT_ID],
            "DOMAIN": ['1b'], 
            "ENSEMBLE_TYPE": ['cached_prior']
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
        account='bdata',
        partition='gpu-2080ti',
        jobtime='8:00:00',
        mem_gb=40,
        job_id_start=1,
        debug_mode=DEBUG_MODE,
        add_name="end",
    )
