from fairseq.moe_checkpoint_utils import merge_expert_and_shared_state
from argparse import ArgumentParser
import os

def find_folders(CHECKPOINTS_TOP_FOLDER, re_string=None):
    NEW_MODEL_FOLDERS = []
    for name, folders, files in os.walk(CHECKPOINTS_TOP_FOLDER):
        regex = re.compile(re_string) if re_string else None
        if 'checkpoint_last.pt' not in files and 'checkpoint_last-shared.pt' not in files:
            continue #no last checkpoint found
        if regex and not regex.match(name):
            continue
        new_name = os.path.relpath(name, CHECKPOINTS_TOP_FOLDER)
        NEW_MODEL_FOLDERS.append(new_name)
    return NEW_MODEL_FOLDERS

def add_args():
    parser = ArgumentParser()
    parser.add_argument('--old-folder', type=str)
    parser.add_argument('--new-folder', type=str)
    parser.add_argument('--subfolder', type=str)
    parser.add_argument('--phase-one-ratio', type=float)
    parser.add_argument('--domain-id', type=int)
    return parser.parse_args()

def main(CHECKPOINTS_TOP_FOLDER, NEW_MODEL_TOP_FOLDER, subfolder, phase_one_ratio, domain_id):
    import shutil, torch
    old_folder = os.path.join(CHECKPOINTS_TOP_FOLDER, subfolder)
    files = [f for f in os.listdir(old_folder) if os.path.isfile(os.path.join(old_folder, f))]
    checkpoint_update_ids = [f[:-3].split('-')[0] for f in files]
    update_nums = [int(f.split("_")[2]) for f in checkpoint_update_ids if f.count('_') == 2]
    max_update_num = max(update_nums)
    src_update_num = int(phase_one_ratio * max_update_num)
    src_checkpoint_update_id = checkpoint_update_ids[min(range(len(update_nums)), key = lambda i: abs(update_nums[i]-src_update_num))]
    if 'checkpoint_last.pt' in files: #dense
        # for domain_id in range(8):
        new_domain_folder_path = os.path.join(NEW_MODEL_TOP_FOLDER, subfolder, f'DOMAIN_ID={domain_id}')
        print(new_domain_folder_path)
        src_filename = os.path.join(old_folder, f'{src_checkpoint_update_id}.pt')
        os.makedirs(new_domain_folder_path, exist_ok=True)
        filename = os.path.join(new_domain_folder_path, 'checkpoint_last.pt')
        shutil.copyfile(src_filename, filename)
    elif 'checkpoint_last-shared.pt' in files: #demix
        # for domain_id in range(8):
        new_domain_folder_path = os.path.join(NEW_MODEL_TOP_FOLDER, subfolder, f'DOMAIN_ID={domain_id}')
        print(new_domain_folder_path)
        expert_path = os.path.join(old_folder, f'{src_checkpoint_update_id}-rank-{domain_id}.pt')
        os.makedirs(new_domain_folder_path, exist_ok=True)
        with open(expert_path, "rb") as f:
            expert_state = torch.load(f, map_location=torch.device("cpu"))
        with open(re.sub('rank-[0-9]+', 'shared', expert_path), "rb") as f:
            shared_state = torch.load(f, map_location=torch.device("cpu"))
        state = moe_checkpoint_utils.merge_expert_and_shared_state(expert_state, shared_state)
        filename = os.path.join(new_domain_folder_path, 'checkpoint_last.pt')
        with open(filename, 'wb') as f:
            torch.save(state, filename)


if __name__=='__main__':
    args = add_args()
    main(args.old_folder, args.new_folder, args.subfolder, args.phase_one_ratio, args.domain_id)