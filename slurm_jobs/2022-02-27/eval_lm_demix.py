from slurm_jobs.slurm_job import run_grid
import os

SWEEP_NAME = "eval_demix_gpt3_medium"
DEBUG_MODE = False
name_keys = []
NUM_GPUS = 8

DATA_BIN='/gscratch/zlab/margsli/gitfiles/demix-data/data-bin'

ROOT_MODEL_FOLDER = '/gscratch/zlab/margsli/demix-checkpoints/models'
MODEL_FOLDER_PREFIX = 'demix_gpt3_medium_with_checkpoints/demix_8_GPUs_transformer_lm_gpt3_medium_tutorial'
CHECKPOINT_IDS = 'best,best,best,best,best,best,best,best'

grids = {
    SWEEP_NAME: {
        'fixed_args': '',
        'positional_args': {            
            "NUM_GPUS": [NUM_GPUS],
            "DATA_BIN": [DATA_BIN],
            "ROOT_MODEL_FOLDER": [ROOT_MODEL_FOLDER],
            "MODEL_FOLDER_PREFIX": [MODEL_FOLDER_PREFIX],
            "CHECKPOINT_IDS": [CHECKPOINT_IDS],
            "DOMAIN": ['cord19-redo', 'github_redo', 'qasper', 'anonymized_tweets_redo', 'anonymized_yelp_reviews_redo', 'anonymized_latest_news_redo', 'legal_contracts', 'gutenberg', '1b', 'anonymized_openwebtext', 'anonymized_realnews', 'anonymized_reviews', 'cs', 'legal', 'med', 'reddit'], 
            "ENSEMBLE_TYPE": ['cached_prior'],
            "MODEL_TYPE": ['demix'],
            "GENERALIST_MODEL": ["None"],
            "TOP_K": [8],
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
        jobtime='8:00:00',
        mem_gb=40,
        job_id_start=1,
        debug_mode=DEBUG_MODE,
        add_name="end",
    )
