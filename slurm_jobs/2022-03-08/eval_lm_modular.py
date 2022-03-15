from slurm_jobs.slurm_job import run_grid
import os

SWEEP_NAME = "eval_modular_on_domains"
DEBUG_MODE = False
DRY_MODE = False
name_keys = []
NUM_GPUS = 1

DATA_BIN='/gscratch/zlab/margsli/gitfiles/demix-data/data-bin'

ROOT_MODEL_FOLDER = '/gscratch/zlab/margsli/demix-checkpoints/models/_dense_1gpu_domains_2'
CHECKPOINT_IDS =  '3_138000,1_138000,1_138000,1_138000,1_138000,1_138000,1_138000,1_138000'.split(',')
# DOMAINS = ['1b','anonymized_openwebtext','anonymized_realnews','anonymized_reviews','cs','legal','med','reddit']
DOMAINS = ['1b','anonymizedopenwebtext','anonymizedrealnews','anonymizedreviews','cs','legal','med','reddit']
MODEL_FOLDERS = [f'dense_1gpu_domains_2_DOMAIN={DOMAINS[i]} {CHECKPOINT_IDS[i]}' for i in range(1,4)]

grids = {
    SWEEP_NAME: {
        'fixed_args': '',
        'positional_args': {
            "DATA_BIN": [DATA_BIN],
            "ROOT_MODEL_FOLDER": [ROOT_MODEL_FOLDER],
            "MODEL_FOLDER": MODEL_FOLDERS,
            # "CHECKPOINT_ID": [CHECKPOINT_ID],
            "SPLIT": ['test'],
            "DOMAIN_ID": [i for i in range(16)], 
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
        account='bdata',
        partition='gpu-2080ti',
        jobtime='2:00:00',
        mem_gb=40,
        job_id_start=1,
        debug_mode=DEBUG_MODE,
        dry_mode=DRY_MODE,
        # add_name="end",
    )
