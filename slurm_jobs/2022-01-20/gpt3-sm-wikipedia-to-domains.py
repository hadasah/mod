from slurm_jobs.slurm_job import run_grid
import os

DEBUG_MODE = False
name_keys = []

TOKENS_PER_SAMPLE=1024
BATCH_SIZE=2
LOG_INTERVAL=50
KEEP_INTERVAL_UPDATES=-1
LR=5e-5
CLIP_NORM=0.1
UPDATE_FREQ=8
NUM_STEPS=300000
SAVE_INTERVAL_UPDATES=3000
VALIDATION_INTERVAL=3000
NUM_WARMUP_STEPS=(NUM_STEPS * 8)//100

DOMAIN=cs
CKPT_UPDATES=
DATA_DIR=/gscratch/zlab/margsli/gitfiles/demix-data/data-bin/{DOMAIN}
NUM_GPUS=1
PORT=12347
ARCH=transformer_lm_gpt3_small
EXPERIMENT=dense
DATA_PATH={DATA_DIR}
PARAMS_TO_FREEZE=self_attn
FILE_SUFFIX=wikipedia_to_{DOMAIN}_freeze_{PARAMS_TO_FREEZE}
SERIALIZATION_DIR=(pwd)/models/dense_gpt3_small_wikipedia_to_{DOMAIN}_freeze_{PARAMS_TO_FREEZE}
NUM_NODES=1
INIT_MODEL=/gscratch/zlab/margsli/gitfiles/demix/models/dense_gpt3_small_wikipedia/dense_8_GPUs_transformer_lm_gpt3_small_wikipedia/checkpoint_1_{CKPT_UPDATES}.pt


grids = {
    "wikipedia_to_domains": {
        "fixed_args": ['--memory-efficient-fp16 --no-epoch-checkpoints --skip-invalid-size-inputs-valid-test'],
        "named_args": {
            '--finetune-from-model': [''],
            '--task': ['language_modeling'],
            '--sample-break-mode': ['none'],
            '--log-format': ['simple'],
            '--log-interval': [LOG_INTERVAL],
            '--validate-interval-updates': [VALIDATION_INTERVAL],    
            '--save-interval-updates': [SAVE_INTERVAL_UPDATES], 
            '--keep-interval-updates': [KEEP_INTERVAL_UPDATES],  
            '--arch': [ARCH], 
            '--criterion': ['cross_entropy'],
            '--lr-scheduler': ['polynomial_decay'],
            '--num-workers': [2],
            '--max-sentences': [BATCH_SIZE],
            '--max-sentences-valid': [BATCH_SIZE],
            '--lr': [LR],        
            '--tokens-per-sample': [TOKENS_PER_SAMPLE],
            '--optimizer': ['adam'],
            '--adam-betas': ['(0.9, 0.95)'],
            '--adam-eps': [10e-8],
            '--weight-decay': [0.1],
            '--clip-norm': [CLIP_NORM],     
            '--max-update': [NUM_STEPS],
            '--total-num-update': [NUM_STEPS],
            '--warmup-updates': [NUM_WARMUP_STEPS],
            '--update-freq': [UPDATE_FREQ],
            '--save-dir': ['{SERIALIZATION_DIR}/dense_{NUM_GPUS}_GPUs_{ARCH}_{FILE_SUFFIX}']    
            '--batch-size-valid': [2],                      
            '--wandb-project': [WANDB_PROJECT],
            '--required-batch-size-multiple': [1], 
            '--distributed-world-size': [NUM_GPUS],
            '--distributed-port': [PORT],
            '--ddp-backend': ['no_c10d'],
            '--params-to-freeze': ['self_attn', '']
            '--all-gather-list-size': [32000],
        },
    },
}

for sweep_name, args in grids.items():
    run_grid(
        args['fixed_args'],
        args['named_args'],
        name_keys,
        sweep_name,
        user=os.environ['USER'],
        prefix='python fairseq_cli/train.py',
        gpus=1,
        cpus=5,
        nodes=1,
        account='zlab',
        partition='gpu-rtx6k',
        jobtime='8:00:00',
        mem_gb=40,
        job_id_start=1,
        debug_mode=DEBUG_MODE,
    )
