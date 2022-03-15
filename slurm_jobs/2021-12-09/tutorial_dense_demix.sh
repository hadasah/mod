#!/bin/bash
#SBATCH --job-name=demix-dense
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=zlab
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=5
#SBATCH --mem=320G
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=1

# I use source to initialize conda into the right environment.
source ~/.bashrc

cat $0
echo "--------------------"
cd /gscratch/zlab/margsli/gitfiles/demix

export DATA_DIR=/gscratch/zlab/margsli/gitfiles/demix-data/example_domains
export NUM_GPUS=8
export DISTRIBUTED_PORT=12345
export MODEL=transformer_lm
export EXPERIMENT=dense
export DATA_BIN=${DATA_DIR}/data-bin/
export EXPERIMENT_SUFFIX=tutorial
export SERIALIZATION_DIR=$(pwd)/tutorial_demix_dense_model
bash demix/train.sh $NUM_GPUS $DISTRIBUTED_PORT $MODEL $EXPERIMENT $DATA_BIN $SERIALIZATION_DIR $EXPERIMENT_SUFFIX