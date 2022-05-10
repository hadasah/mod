 import os
 import re


def add_args():
    parser = ArgumentParser()
    parser.add_argument('--regex-name-str', type=str)
    parser.add_argument('--model-folder', type=str)
    parser.add_argument('--exclude-expert', type=str)
    parser.add_argument('--only-use-expert', type=str)
    parser.add_argument('--generalist-model', type=str)
    parser.add_argument('--model-type', type=str)
    parser.add_argument('--checkpoint-ids', type=str)
    parser.add_argument('--target-domain-id', type=int)
    return parser.parse_args()



def main(
    regex_str, model_folder, model_type, checkpoint_ids, generalist_model,
    only_use_expert, exclude_expert, target_domain_id):
    models = []
    exclude_expert = bool(exclude_expert)
    only_use_expert = bool(only_use_expert)
    str_name = regex_str.replace('.*', '')
    eval_name = str_name
    checkpoint_ids = checkpoint_ids.split(',')
    all_runs = os.listdir(model_folder)
    regex = re.compile(regex_str)
    selected_folders = ':'.join(sorted([folder for folder in all_runs if regex.match(folder)]))
    result_folder = os.path.join(model_folder, 'evals', eval_name, str(target_domain_id))
    
    for i in range(8):
        if (
            (not only_use_expert or target_domain_id == i)
            and 
            (not exclude_expert or target_domain_id != i)
        ):
            if model_type == 'demix':
                models.append(f'{model_folder}/checkpoint_{checkpoint_ids[i]}-rank-{i}.pt'
            elif model_type == 'modular':
                models.append(f'{model_folder}/{selected_folders[i]}/checkpoint_{checkpoint_ids[i]}.pt'
    
    if generalist_model != "None":
        models.append(generalist_model)
        
    return result_folder, ':'.join(models)


if __name__=='__main__':
    args = add_args()
    main(args.regex_str, args.model_folder, args.model_type, args.checkpoint_ids, args.generalist_model,
    args.only_use_expert, args.exclude_expert)
