#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import ast
import os
import random
import sys
from copy import copy
from typing import Dict, Optional, Any, List, Tuple, Callable

import numpy as np
import torch
import torch.distributed as dist

from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed_utils import is_master
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer
from omegaconf import DictConfig, OmegaConf

os.environ["NCCL_DEBUG"] = "INFO"

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]



def main(cfg: FairseqConfig) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    if is_master(cfg.distributed_training) and "job_logging_cfg" in cfg:
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    logger.info(cfg)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)
    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in cfg.dataset.valid_subset.split(","):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    assert cfg.criterion, "Please specify criterion to train a model"

    # Build model and criterion
    model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)
    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))


    if torch.distributed.is_initialized() and getattr(cfg.model, "desynchronize", False):
        
        
        if getattr(cfg.model, "sync_type") == "none":
            groups = [[x] for x in range(torch.distributed.get_world_size())]
            
        elif getattr(cfg.model, "sync_type") == "manual":
            gps = getattr(cfg.model, "data_parallel_groups")
            gps = gps.split()
            gps = [gp.split(',') for gp in gps]
            groups = [[int(x) for x in y] for y in gps]

        process_groups = {}
        for group in groups:
            distributed_group = torch.distributed.new_group(group)
            for item in group:
                process_groups[item] = distributed_group


        logger.info(f"Data Parallel Groups: {groups}")
        process_group = process_groups[torch.distributed.get_rank(torch.distributed.group.WORLD)]
        
        if getattr(cfg.model, "untie_parameters"):
            for x, p in model.named_parameters():

                if getattr(cfg.model, "untie_parameters") == "transformer_l2":
                    every_other_layer = [f"layers.{z}" for z in range(0, getattr(cfg.model, "decoder_layers"), 2)]
                    if any(l in x for l in every_other_layer):
                        p.expert = True
                        p.process_group = process_group

                elif getattr(cfg.model, "untie_parameters") == "expert_layers":
                    if "expert_layers" in x:
                        p.expert = True
                        p.process_group = process_group

                elif getattr(cfg.model, "untie_parameters") == "expert_ffn":
                    ffns = ['expert_fc1', 'expert_fc2', 'gate']
                    if any(ffn in x for ffn in ffns):
                        p.expert = True
                        p.process_group = process_group


                elif getattr(cfg.model, "untie_parameters") == "transformer_output":
                    if "layers" in x or 'output_projection' in x:
                        p.expert = True
                        p.process_group = process_group
                
                elif getattr(cfg.model, "untie_parameters") == "input_output":
                    if "embed_tokens" in x or "embed_positions" in x or 'output_projection' in x:
                        p.expert = True
                        p.process_group = process_group
                
                elif getattr(cfg.model, "untie_parameters") == "transformer":
                    if "layers" in x:
                        p.expert = True
                        p.process_group = process_group

                elif getattr(cfg.model, "untie_parameters") == "transformer_l2":
                    every_other_layer = [f"layers.{z}" for z in range(0, getattr(cfg.model, "decoder_layers"), 2)]
                    if any(l in x for l in every_other_layer):
                        p.expert = True
                        p.process_group = process_group

                elif getattr(cfg.model, "untie_parameters") == "feedforward":
                    ffns = ['fc1', 'fc2']
                    if any(ffn in x for ffn in ffns) and 'expert' not in x:
                        p.expert = True
                        p.process_group = process_group
                
                elif getattr(cfg.model, "untie_parameters") == "feedforward_l2":
                    every_other_layer = [f"layers.{z}" for z in range(0, getattr(cfg.model, "decoder_layers"), 2)]
                    ffns = ['fc1', 'fc2']
                    if any(ffn in x for ffn in ffns) and any(l in x for l in every_other_layer):
                        p.expert = True
                        p.process_group = process_group

                elif getattr(cfg.model, "untie_parameters") == "feedforward_top":
                    top_layer = list(range(0, getattr(cfg.model, "decoder_layers")))[-1]
                    top_layer = [f"layers.{top_layer}"]
                    ffns = ['fc1', 'fc2']
                    if any(ffn in x for ffn in ffns) and any(l in x for l in top_layer):
                        p.expert = True
                        p.process_group = process_group

                elif getattr(cfg.model, "untie_parameters") == "all":
                    p.expert = True
                    p.process_group = process_group
                else:
                    raise Exception("value for untie_parameters is bad.")
    

    if getattr(cfg.model, "adaptation", False):
    #     # if not getattr(cfg.model, "untie_parameters"):
    #     #     for layer in model.decoder.layers:
    #     #         layer.add_adapter(1)
    #     #     for x,p in model.named_parameters():
    #     #         if not 'adapter' in x:
    #     #             p.requires_grad = False
         ffns = ['fc1', 'fc2']
         for x,p in model.named_parameters():
            if not any(ffn in x for ffn in ffns):
                 p.requires_grad = False
            else:
                 p.requires_grad = True

    if getattr(cfg.model, "params_to_freeze", "None") != "None": 
        params_to_freeze = [s for s in getattr(cfg.model, "params_to_freeze", "").split(",") if len(s) > 0]
        dec_layers = getattr(cfg.model, "decoder_layers")
        params_to_freeze_shortnames = {
            "first_half_self_attn": ['.' + str(i) + '.self_attn' for i in range(0, dec_layers//2)], 
            "second_half_self_attn": ['.' + str(i) + '.self_attn' for i in range(dec_layers//2, dec_layers)], 
            "mid_half_self_attn": ['.' + str(i) + '.self_attn' for i in range(dec_layers//4, 3*dec_layers//4)],
            "all_first_half": ['.' + str(i) + '.' for i in range(0, dec_layers//2)],
            "all_second_half": ['.' + str(i) + '.' for i in range(dec_layers//2, dec_layers)],
            "all_ffn": ['.fc1', '.fc2']
        }
        new_params_to_freeze = copy(params_to_freeze)
        for ptf in params_to_freeze:
            if ptf in params_to_freeze_shortnames:
                new_params_to_freeze.remove(ptf)
                new_params_to_freeze.extend(params_to_freeze_shortnames[ptf])
        for x, p in model.named_parameters():
            if any(p_name in x for p_name in new_params_to_freeze):
                p.requires_grad = False
            else:
                p.requires_grad = True

    logger.info(
        "num. shared model params: {} (num. trainable: {})".format(
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False) ),
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False) and p.requires_grad),
        )
    )
    
    logger.info(
        "num. expert model params: {} (num. trainable: {})".format(
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False) ),
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False) and p.requires_grad),
        )
    )
    
    

    # (optionally) Configure quantization
    if cfg.common.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=cfg.common.quantization_config_path,
            max_epoch=cfg.optimization.max_epoch,
            max_update=cfg.optimization.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if cfg.common.model_parallel_size == 1:
        trainer = Trainer(cfg, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(cfg, task, model, criterion)

    logger.info(
        "training on {} devices (GPUs/TPUs)".format(
            cfg.distributed_training.distributed_world_size
        )
    )
    logger.info(
        "max tokens per GPU = {} and batch size per GPU = {}".format(
            cfg.dataset.max_tokens,
            cfg.dataset.batch_size,
        )
    )
    # reset the checkpoint suffix again after setting the process groups (e.g. for demix)
    print('cfg.checkpoint.checkpoint_suffix', cfg.checkpoint.checkpoint_suffix)
    if getattr(cfg.model, "desynchronize", False) or getattr(cfg.model, "moe_freq", 0) > 0 and getattr(cfg.model, "moe_expert_count", 0) > 0:
        cfg.checkpoint.checkpoint_suffix = f"-rank-{cfg.distributed_training.distributed_rank}"
        print('cfg.checkpoint.checkpoint_suffix', cfg.checkpoint.checkpoint_suffix)


    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )

    if getattr(cfg.model, "adaptation", False):
        for x,p in model.named_parameters():
            if hasattr(p, "expert"):
                delattr(p, "expert")
                delattr(p, "process_group")

    max_epoch = cfg.optimization.max_epoch or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()
    while epoch_itr.next_epoch_idx <= max_epoch:
        if lr <= cfg.optimization.stop_min_lr:
            logger.info(
                f"stopping training because current learning rate ({lr}) is smaller "
                "than or equal to minimum learning rate "
                f"(--stop-min-lr={cfg.optimization.stop_min_lr})"
            )
            break

        # train for one epoch
        valid_losses, should_stop = train(cfg, trainer, task, epoch_itr)
        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
        )
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))


def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    cfg.checkpoint.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(
    cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > cfg.dataset.curriculum),
    )
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    if cfg.common.tpu:
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        wandb_project=(
            cfg.common.wandb_project
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
        ),
        wandb_entity=(
            cfg.common.wandb_entity
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        azureml_logging=(
            cfg.common.azureml_logging
            if distributed_utils.is_master(cfg.distributed_training)
            else False
        ),
    )
    progress.update_config(_flatten_config(cfg))

    trainer.begin_epoch(epoch_itr.epoch)

    valid_subsets = cfg.dataset.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    for i, samples in enumerate(progress):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):
            log_output = trainer.train_step(samples)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def _flatten_config(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config


def validate_and_save(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    valid_subsets: List[str],
    end_of_epoch: bool,
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf

    # Stopping conditions (and an additional one based on validation loss later
    # on)
    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"num_updates: {num_updates} >= max_update: {max_update}"
        )

    training_time_hours = trainer.cumulative_training_time() / (60 * 60)
    if (
        cfg.optimization.stop_time_hours > 0
        and training_time_hours > cfg.optimization.stop_time_hours
    ):
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"cumulative_training_time: {training_time_hours} > "
            f"stop_time_hours: {cfg.optimization.stop_time_hours} hour(s)"
        )

    do_save = (
        (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
        or should_stop
        or (
            cfg.checkpoint.save_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.checkpoint.save_interval_updates == 0
            and num_updates >= cfg.dataset.validate_after_updates
        )
    )
    do_validate = (
        (not end_of_epoch and do_save)  # validate during mid-epoch saves
        or (end_of_epoch and epoch_itr.epoch % cfg.dataset.validate_interval == 0)
        or should_stop
        or (
            cfg.dataset.validate_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.dataset.validate_interval_updates == 0
        )
    ) and not cfg.dataset.disable_validation
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)

    should_stop |= should_stop_early(cfg, valid_losses[0])

    # Save checkpoint
    if do_save or should_stop:
        checkpoint_utils.save_checkpoint(
            cfg.checkpoint, trainer, epoch_itr, valid_losses[0], training_finished=should_stop,
        )

    trainer.reset_dummy_batch(epoch_itr.first_batch)
    return valid_losses, should_stop


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    subsets: List[str],
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    for subset in subsets:
        
        if torch.distributed.is_initialized():
            logger.info('begin validation on "{}" subset on rank {}'.format(
                subset, torch.distributed.get_rank()))
        else:
            logger.info('begin validation on "{}" subset'.format(
                subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(
            shuffle=False, set_dataset_epoch=False  # use a fixed valid set
        )
        if cfg.common.tpu:
            itr = utils.tpu_data_loader(itr)
        
        if torch.distributed.is_initialized():
            logger.info('got valid iterator on "{}" subset on rank {}'.format(
                    subset,
                    torch.distributed.get_rank()
                )
            )
        else:
            logger.info('got valid iterator on "{}"'.format(
                    subset
                )
            )

        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir
                if torch.distributed.is_initialized() and distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project
                if torch.distributed.is_initialized() and distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            wandb_run_name=os.environ.get(
                "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
            ),
            wandb_entity=(
                cfg.common.wandb_entity
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
        )
        logger.info('Begin looping over validation "{}" subset with length "{}"'.format(subset, len(progress)))

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                trainer.valid_step(sample)
        # log validation stats
        stats = get_valid_stats(cfg, trainer, agg.get_smoothed_values())
        progress.print(stats, tag=subset, step=trainer.get_num_updates())
        valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(
    cfg: DictConfig, trainer: Trainer, stats: Dict[str, Any]
) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[cfg.checkpoint.best_checkpoint_metric],
        )
    return stats


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)


if __name__ == "__main__":
    cli_main()
