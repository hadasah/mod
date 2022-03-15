from slurm_jobs.slurm_job import run_grid
import os

SWEEP_NAME = "modular_gpt3_medium_8gpuday"
NUM_GPUS = 1
DEBUG_MODE = False
name_keys = []

grids = {
    SWEEP_NAME: {
        'fixed_args': '',
        'positional_args': {
            "SWEEP_NAME": [SWEEP_NAME],
            "NUM_GPUS": [NUM_GPUS],
            "ARCH": ['transformer_lm_gpt3_medium'],
            "EXPERIMENT": ['dense'],
            "DOMAIN": ['1b', 'anonymized_openwebtext', 'anonymized_realnews', 'anonymized_reviews', 'cs', 'legal', 'med',  'reddit'], 
            "TOP_LEVEL_SERIALIZATION_DIR": ['/gscratch/zlab/margsli/demix-checkpoints/models'],
            "NUM_STEPS": [6000],
            "UPDATE_FREQ": [32],
            "LR": [5e-4, 1e-3, 5e-3, 2e-3],
            "WANDB_PROJECT": ['modular_experiments'],
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
        prefix='bash /mmfs1/gscratch/zlab/margsli/gitfiles/demix/demix/train_from_scratch.sh',
        gpus=NUM_GPUS,
        cpus=5,
        nodes=1,
        account='cse',
        partition='gpu-rtx6k',
        jobtime='24:00:00',
        mem_gb=40,
        job_id_start=1,
        debug_mode=DEBUG_MODE,
        add_name="end",
    )
