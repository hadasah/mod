import torch
from pathlib import Path
from tqdm.auto import tqdm, trange
import argparse
import numpy as np
from fairseq.models.transformer_lm import TransformerLanguageModel
import math
from fairseq_cli.eval_lm import eval_lm
import logging
import math
import os
import pandas as pd
import sys
from argparse import Namespace
from typing import Iterable, List, Optional

import torch
import fairseq
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter
from fairseq.sequence_scorer import SequenceScorer
from omegaconf import DictConfig



def load_expert(path, file="checkpoint_last.pt"):
    task = tasks.setup_task("multidomain_language_modeling")
    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        [path],
        arg_overrides=None,
        suffix="checkpoint_last.pt",
        strict=0,
        num_shards=0,
        task=task,
        moe_freq=moe_freq,
        desynchronize=cfg.common_eval.is_moe and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1,
        partial_load=cfg.common_eval.partial_load
    )


    return models, model_args, tasks
    
def average(models, weights=None, topk=-1, uniform=False):
    if uniform:
        weights = [1/len(models)] * len(models)
    elif topk == 1:
        weights = [1.0]
    if topk > 0:
        sorted_models = list(sorted(zip(models, weights), key=lambda x: x[1],reverse=True))[:topk]
        models = [x[0] for x in sorted_models]
        weights = [x[1] for x in sorted_models]
    state_dicts = [model['model'] for model in models]
    with torch.no_grad():
        merged = {}
        for key in state_dicts[0]:
            merged[key] = torch.sum(torch.stack([sd[key] * weight for sd, weight in zip(state_dicts, weights)]), axis=0)
        
        return merged

# demix_expert = TransformerLanguageModel.from_pretrained('/checkpoint/suching/mod/_modular_gpt3_small_80K/modular_gpt3_small_80K_LR=0.0005/MODEL=transformerlmgpt3small_DOMAINID=6_LOADFROMSTEP=24000_RESETITEMS=dataloader_UPDATEFREQ=32_LR=0.0005/', data_name_or_path='/private/home/suching/raw_data/demix_scale/data-bin', checkpoint_file='checkpoint_last.pt', suffix='', moe_freq=0, desynchronize=True, bpe='gpt2')
# demix_expert.models[0].parameters

def score_expert(expert, data):




    scores = expert.score(data)
    score_sum = 0.0
    count = 0
    for score in scores:
        score_sum += score['positional_scores'].float().sum().cpu()
        count += score['positional_scores'].numel()
    # import pdb; pdb.set_trace()
    # score_sum = torch.cat([x['score'].unsqueeze(0) for x in scores])
    ppl = 2 ** (-score_sum / count / math.log(2) if count > 0 else 0)
    return ppl


def average_soup(soup, output_dir, weights=None, uniform=False, topk=-1, save_path='checkpoint_last.pt', save=False):
    # if avg:
    averaged_soup = soup[0].copy()
    averaged_soup['model'] = average(soup, weights=weights, uniform=uniform, topk=topk)
    if save:
        torch.save(averaged_soup, output_dir / save_path)
    else:
        return averaged_soup
    # else:
    #     averaged_soup = experts[int(np.argmax(weights))].copy()
    #     # merged_expert['model'] = experts[]
    
    return averaged_soup

def argmax_soup(soup, weights, output_dir, save_path='checkpoint_last.pt'):
    merged_expert = experts[int(np.argmax(weights))].copy()
    torch.save(merged_expert, output_dir / save_path)
    
def evaluate_soup(data, output_dir, save_path='checkpoint_last.pt'):
    averaged_expert = load_expert(output_dir, save_path)
    ppl = score_expert(averaged_expert, data)
    return ppl

def main(model_dir, dev, output_dir, load_from_step, weights, topk, uniform, additional_domains, avg, greedy_soup):
        
    

    # weights = [0.0000028327,0.3556927243,0.1098412628,0.4213291476,0.0000181846,0.000215643,0.0000005297,0.1128996771]
    with open(dev, 'r') as f:
        dev_set = [x.strip() for x in f.readlines()][:10]
    

    experts = list(Path(model_dir).glob(f'MODEL*{load_from_step}*0.0005'))
    additional_experts = []

    if additional_domains:

        for domain in additional_domains:
            candidates = list(Path('/checkpoint/suching/mod/_adaptation_transformer_lm_gpt3_small/adaptation_transformer_lm_gpt3_small_LR=0.0005/').glob('MODEL*-1*True*'))
            additional_experts.append([candidate for candidate in candidates if domain in str(candidate)][0])

    #additional_experts = list(Path('/checkpoint/suching/mod/_adaptation_transformer_lm_gpt3_small/adaptation_transformer_lm_gpt3_small_LR=0.0005/').glob('MODEL*-1*True*'))
    experts = experts + additional_experts 
    print(experts)
    expert_pool = [(e / 'checkpoint_last.pt', torch.load(e / 'checkpoint_last.pt')) for e in tqdm(experts)]



    averaged_expert = None
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)
    if greedy_soup:
        # initial_soup = average_soup([x[1] for x in expert_pool], output_dir, weights=weights, save=True, save_path='checkpoint_last_tmp.pt')
        soup = []
        soup_paths = []
        best_ppl = np.inf
        # evaluate_soup(dev_set, output_dir, save_path='checkpoint_last_tmp.pt')
        save_path = 'checkpoint_last_tmp.pt'
        pbar = trange(100)
        for _ in pbar:
            random_idx = np.random.choice(range(len(experts)),  p=weights)
            random_expert_path, random_expert = expert_pool[random_idx]
            averaged_expert = average_soup(soup + [random_expert], output_dir, uniform=True, save_path=save_path, save=True)
            ppl = evaluate_soup(dev_set, output_dir, save_path=save_path)
            if ppl >= best_ppl:
                continue
            else:
                soup.append(random_expert)
                soup_paths.append(random_expert_path)
                best_average = averaged_expert
                best_ppl = ppl
            pbar.set_description(f"best ppl: {best_ppl}, len of soup: {len(soup)}")
        print(f"best soup: {soup_paths}")
        torch.save(best_average, output_dir / 'checkpoint_last.pt')
    else:
        save_path = 'checkpoint_last.pt'
        average_soup([x[1] for x in expert_pool], output_dir, weights=weights, uniform=uniform, topk=topk, save=True, save_path=save_path)
        # print(evaluate_soup(dev_set, output_dir, save_path=save_path))


    # merged_expert = TransformerLanguageModel.from_pretrained(output_dir,
    #                                         data_name_or_path='/private/home/suching/raw_data/demix_scale/data-bin',
    #                                         checkpoint_file='checkpoint_last.pt',
    #                                         suffix='',
    #                                         moe_freq=0,
    #                                         device=0,
    #                                         desynchronize=True,
    #                                         bpe='gpt2')
    # ppl = score_expert(merged_expert, lines)
    
    
    # import pdb; pdb.set_trace()
    # return merged_expert, ppl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-dir", type=str)
    parser.add_argument("--dev", type=str)
    parser.add_argument("--load-from-step", type=int)
    parser.add_argument("--weights", type=str)
    parser.add_argument("--topk", type=int)
    parser.add_argument("--uniform", action='store_true')
    parser.add_argument("--additional-domains", type=str, nargs="+")
    parser.add_argument("--average", action='store_true')
    parser.add_argument("--greedy-soup", action='store_true')

    args = parser.parse_args()
    if args.weights:
        weights = [float(x) for x in args.weights.split(',')]
    else:
        weights = args.weights

    main(args.model_dir, args.dev, args.output_dir, args.load_from_step, weights, args.topk, args.uniform, args.additional_domains, args.average, args.greedy_soup)
