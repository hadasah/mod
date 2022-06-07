---
license: other
tags:
- generated_from_trainer
model-index:
- name: test_opt
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# test_opt

This model is a fine-tuned version of [facebook/opt-350m](https://huggingface.co/facebook/opt-350m) on an unknown dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 16
- eval_batch_size: 4
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Framework versions

- Transformers 4.20.0.dev0
- Pytorch 1.9.0+cu102
- Datasets 1.15.1
- Tokenizers 0.11.6
