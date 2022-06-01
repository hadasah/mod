import torch
from pathlib import Path
from tqdm.auto import tqdm
import argparse

def average(models, weights=None):
        state_dicts = [model['model'] for model in models]
        with torch.no_grad():
            merged = {}
            for key in state_dicts[0]:
                merged[key] = torch.sum(torch.stack([sd[key] * weight for sd, weight in zip(state_dicts, weights)]), axis=0)
            return merged

# demix_expert = TransformerLanguageModel.from_pretrained('/checkpoint/suching/mod/_modular_gpt3_small_80K/modular_gpt3_small_80K_LR=0.0005/MODEL=transformerlmgpt3small_DOMAINID=6_LOADFROMSTEP=24000_RESETITEMS=dataloader_UPDATEFREQ=32_LR=0.0005/', data_name_or_path='/private/home/suching/raw_data/demix_scale/data-bin', checkpoint_file='checkpoint_last.pt', suffix='', moe_freq=0, desynchronize=True, bpe='gpt2')
# demix_expert.models[0].parameters

def main(output_dir, weights, additional_domains):
        
    

    # weights = [0.0000028327,0.3556927243,0.1098412628,0.4213291476,0.0000181846,0.000215643,0.0000005297,0.1128996771]


    experts = list(Path('/checkpoint/suching/mod/_modular_gpt3_small_80K/modular_gpt3_small_80K_LR=0.0005').glob('MODEL*24000*0.0005'))
    additional_experts = []
    for domain in additional_domains:
        candidates = list(Path('/checkpoint/suching/mod/_adaptation_transformer_lm_gpt3_small/adaptation_transformer_lm_gpt3_small_LR=0.0005/').glob('MODEL*-1*True*'))
        additional_experts.append([candidate for candidate in candidates if domain in str(candidate)][0])
    #additional_experts = list(Path('/checkpoint/suching/mod/_adaptation_transformer_lm_gpt3_small/adaptation_transformer_lm_gpt3_small_LR=0.0005/').glob('MODEL*-1*True*'))
    experts = experts + additional_experts 
    print(experts)

    experts = [torch.load(e / 'checkpoint_last.pt') for e in tqdm(experts)]


    merged_expert = experts[0].copy()

    merged_expert['model'] = average(experts, weights=weights)

    torch.save(merged_expert, output_dir / 'checkpoint_last.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--weights", type=str)
    parser.add_argument("--additional-domains", type=str, nargs="+")
    args = parser.parse_args()
    main(args.output_dir, [float(x) for x in args.weights.split(',')],args.additional_domains)
