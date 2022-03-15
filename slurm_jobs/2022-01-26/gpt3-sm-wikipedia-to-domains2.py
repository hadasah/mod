from slurm_jobs.slurm_job import run_grid
import os

SWEEP_NAME = "wikipedia_to_domains_a40test"
NUM_GPUS = 1
DEBUG_MODE = True
name_keys = []

grids = {
    SWEEP_NAME: {
        'fixed_args': '',
        'positional_args': {
            "SWEEP_NAME": [SWEEP_NAME],
            "NUM_GPUS": [NUM_GPUS],
            "ARCH": ['transformer_lm_gpt3_small'],
            "EXPERIMENT": ['dense'],
            "DOMAIN": ['1b', 'cs', 'legal', 'med', 'openwebtext', 'realnews', 'reddit', 'reviews'], #'anonymized_openwebtext', 'anonymized_realnews', 'anonymized_reviews', 
            "TOP_LEVEL_SERIALIZATION_DIR": ['/gscratch/zlab/margsli/demix-checkpoints/models'],
            "WANDB_PROJECT": ['gpt3_experiments'],
            "FREEZE_PARAMS": ["all_second_half", ".embed_", "second_half_self_attn"],
            "INIT_MODEL_FOLDER": ["/gscratch/zlab/margsli/gitfiles/demix/models/dense_gpt3_small_wikipedia_all/ckpt"],
            "CKPT": [6000, 12000, 36000, 120000],
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
        partition='gpu-a40',
        jobtime='8:00:00',
        mem_gb=40,
        job_id_start=1,
        debug_mode=DEBUG_MODE,
        add_name="end",
    )
