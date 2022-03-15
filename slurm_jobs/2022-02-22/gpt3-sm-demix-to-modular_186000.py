from slurm_jobs.slurm_job import run_grid
import os

SWEEP_NAME = "demix_186000_to_modular_4day"
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
            "EXPERIMENT": ['demix'],
            "DOMAIN_ID": [i for i in range(0, 8)], 
            "TOP_LEVEL_SERIALIZATION_DIR": ['/gscratch/zlab/margsli/demix-checkpoints/models'],
            "WANDB_PROJECT": ['modular_experiments'],
            "PARAMS_TO_FREEZE": ["None"],
            # "PARAMS_TO_FREEZE": ["self_attn"],
            "INIT_MODEL_FOLDER": ["/gscratch/zlab/margsli/gitfiles/demix/models/demix_gpt3_small_with_checkpoints/demix_8_GPUs_transformer_lm_gpt3_small_tutorial"],
            "CKPT": [186000],
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
        jobtime='24:00:00',
        mem_gb=40,
        job_id_start=1,
        debug_mode=DEBUG_MODE,
        add_name="end",
    )
