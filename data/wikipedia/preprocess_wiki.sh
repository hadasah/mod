#!/bin/bash
#SBATCH --job-name==data-process
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=bdata
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=210G
#SBATCH --time=48:00:00

# I use source to initialize conda into the right environment.
source ~/.bashrc

cat $0
echo "--------------------"

cd /gscratch/zlab/margsli


# for NUM in 0 1 2 3 4 5 6 7 8 9; do \
# fairseq-preprocess \
#     --only-source \
#     --srcdict gpt2_bpe/dict.txt \
#     --trainpref data/wikipedia/bpe/wikipedia-en-${NUM}.bpe \
#     --destdir data-bin/wikipedia-en-${NUM} \
#     --workers 60
# done

fairseq-preprocess \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref data/wikipedia/bpe/wikipedia-en-all.bpe \
    --validpref data/wikipedia/bpe/test-wikipedia-en.bpe \
    --testpref data/wikipedia/bpe/valid-wikipedia-en.bpe \
    --destdir data-bin/wikipedia-en/wikipedia-en-all \
    --workers 60

