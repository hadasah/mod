from slurm_jobs.slurm_job import run_grid
import os

SWEEP_NAME = "eval_dense_4day"
DEBUG_MODE = False
name_keys = []
NUM_GPUS = 1

DATA_BIN='/gscratch/zlab/margsli/gitfiles/demix-data/data-bin'

ROOT_MODEL_FOLDER = '/gscratch/zlab/margsli/gitfiles/demix/models'
MODEL_FOLDER = 'dense_gpt3_small_with_checkpoints/dense_8_GPUs_transformer_lm_gpt3_small_tutorial'
CHECKPOINT_ID = 'best'

grids = {
    SWEEP_NAME: {
        'fixed_args': '',
        'positional_args': {
            "DATA_BIN": [DATA_BIN],
            "ROOT_MODEL_FOLDER": [ROOT_MODEL_FOLDER],
            "MODEL_FOLDER": [MODEL_FOLDER],
            "CHECKPOINT_ID": [CHECKPOINT_ID],
            "SPLIT": ['test'],
            "DOMAIN": ['cord19-redo', 'github_redo', 'qasper', 'anonymized_tweets_redo', 'anonymized_yelp_reviews_redo', 'anonymized_latest_news_redo', 'legal_contracts', 'gutenberg', '1b', 'anonymized_openwebtext', 'anonymized_realnews', 'anonymized_reviews', 'cs', 'legal', 'med', 'reddit'], 
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
        prefix='bash /mmfs1/gscratch/zlab/margsli/gitfiles/demix/demix/eval_pipeline.sh',
        gpus=NUM_GPUS,
        cpus=5,
        nodes=1,
        account='zlab',
        partition='gpu-rtx6k',
        jobtime='8:00:00',
        mem_gb=40,
        job_id_start=1,
        debug_mode=DEBUG_MODE,
        add_name="end",
    )
