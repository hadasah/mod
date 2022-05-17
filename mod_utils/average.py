import torch
from pathlib import Path

def average(models, weights=None):
    state_dicts = [model['model'] for model in models]
    with torch.no_grad():
        merged = {}
        for key in state_dicts[0]:
            merged[key] = torch.sum(torch.stack([sd[key] * weight for sd, weight in zip(state_dicts, weights)]), axis=0)
        return merged

experts = list(Path('/checkpoint/suching/mod/_modular_gpt3_small_80K/modular_gpt3_small_80K_LR=0.0005').glob('MODEL*56000*0.0005'))
experts = [torch.load(e / 'checkpoint_last.pt') for e in experts if 'DOMAINID=4' in e.name or 'DOMAINID=6' in e.name]

print(len(experts))

merged_expert = experts[0].copy()

merged_expert['model'] = average(experts)

torch.save(merged_expert, '/checkpoint/suching/fp16/merged_mod/checkpoint_best.pt')

