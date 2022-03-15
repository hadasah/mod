from slurm_jobs.slurm_job import run_grid
import os

SWEEP_NAME = "wikipedia_to_domains_with_checkpoints"
NUM_GPUS = 1
DEBUG_MODE = False
name_keys = []

grids = {
    SWEEP_NAME: {
        'fixed_args': '',
        'positional_args': {
            "SWEEP_NAME": [SWEEP_NAME],
            "NUM_GPUS": [NUM_GPUS],
            "ARCH": ['transformer_lm_gpt3_small'],
            "EXPERIMENT": ['dense'],
            # "DOMAIN": ['1b', 'anonymized_openwebtext', 'anonymized_realnews', 'anonymized_reviews', 'cs', 'legal', 'med', 'reddit'], 
            "DOMAIN": [i for i in range(8)],
            "TOP_LEVEL_SERIALIZATION_DIR": ['/gscratch/zlab/margsli/demix-checkpoints/models'],
            "WANDB_PROJECT": ['modular_experiments'],
            "FREEZE_PARAMS": ["None"],
            "INIT_MODEL_FOLDER": ["/gscratch/zlab/margsli/gitfiles/demix/models/dense_gpt3_small_wikipedia_all/dense_8_GPUs_transformer_lm_gpt3_small_wikipedia_all"],
            "CKPT": [120000],
            "CONTINUE_TRAIN": ["False"],
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
        prefix='bash /mmfs1/gscratch/zlab/margsli/gitfiles/demix/demix/downstream_train.sh',
        gpus=NUM_GPUS,
        cpus=5,
        nodes=1,
        account='zlab',
        partition='gpu-rtx6k',
        jobtime='48:00:00',
        mem_gb=20,
        job_id_start=1,
        debug_mode=DEBUG_MODE,
        add_name="end",
    )
