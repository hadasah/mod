from fairseq.moe_checkpoint_utils import merge_expert_and_shared_state
from argparse import ArgumentParser
import os
import re


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
    parser.add_argument('--new-subfolder', type=str)
    parser.add_argument('--load-from-step', type=int)
    parser.add_argument('--domain-id', type=str)
    return parser.parse_args()

def main(CHECKPOINTS_TOP_FOLDER, NEW_MODEL_TOP_FOLDER, subfolder, new_subfolder, load_from_step, domain_id):
    import shutil, torch
    # is_master_process = (not torch.distributed.is_initialized()) or (
    #     torch.distributed.is_initialized() and torch.distributed.get_rank() == 0
    # )
    # if not is_master_process:
    #     return
    distributed_rank = int(os.environ['SLURM_PROCID']) if torch.distributed.is_initialized() else 0
    if distributed_rank != 0:
        return
    if not new_subfolder:
        new_subfolder = subfolder
    old_folder = os.path.join(CHECKPOINTS_TOP_FOLDER, subfolder)
    files = [f for f in os.listdir(old_folder) if os.path.isfile(os.path.join(old_folder, f))]
    checkpoint_update_ids = list(set([f[:-3].split('-')[0] for f in files]))
    print('checkpoint_update_ids', checkpoint_update_ids)
    checkpoint_update_ids = [f for f in checkpoint_update_ids if f.count('_') == 2]
    print('checkpoint_update_ids', checkpoint_update_ids)
    # update_nums = [int(f.split("_")[2]) for f in checkpoint_update_ids]
    # print(update_nums)
    # max_update_num = max(update_nums)
    # print('max_update_num', max_update_num)
    # src_update_num = phase_one_ratio
    # print('src_update_num', src_update_num)
    # sort_factor = 1
    # if phase_one_ratio > 0.5:
    #     sort_factor = -1
    # zipped_name_and_num = [(a, b) for (a, b) in zip(update_nums, checkpoint_update_ids)]
    # zipped_name_and_num.sort(key=lambda i: sort_factor * i[0])
    # print('zipped_name_and_num', zipped_name_and_num)
    # src_checkpoint_update_id = zipped_name_and_num[min(range(len(zipped_name_and_num)), key=lambda i: abs(zipped_name_and_num[i][0]-src_update_num))][1]
    if load_from_step == -1:
        src_checkpoint_update_id = f"checkpoint_last"
    else:
        src_checkpoint_update_id = f"checkpoint_1_{load_from_step}"
    # print('src_checkpoint_update_id', src_checkpoint_update_id)
    if 'checkpoint_last.pt' in files: #dense
        # for domain_id in range(8):
        new_domain_folder_path = os.path.join(NEW_MODEL_TOP_FOLDER, new_subfolder)
        print('new_domain_folder_path', new_domain_folder_path)
        src_filename = os.path.join(old_folder, f'{src_checkpoint_update_id}.pt')
        print('src_filename', src_filename)
        os.makedirs(new_domain_folder_path, exist_ok=True)
        filename = os.path.join(new_domain_folder_path, 'checkpoint_last.pt')
        shutil.copyfile(src_filename, filename)
    elif 'checkpoint_last-shared.pt' in files: #demix
        # for domain_id in range(8):
        new_domain_folder_path = os.path.join(NEW_MODEL_TOP_FOLDER, new_subfolder)
        print('new_domain_folder_path', new_domain_folder_path)
        expert_path = os.path.join(old_folder, f'{src_checkpoint_update_id}-rank-{domain_id}.pt')
        print('expert_path', expert_path)
        os.makedirs(new_domain_folder_path, exist_ok=True)
        with open(expert_path, "rb") as f:
            expert_state = torch.load(f, map_location=torch.device("cpu"))
        with open(re.sub('rank-[0-9]+', 'shared', expert_path), "rb") as f:
            shared_state = torch.load(f, map_location=torch.device("cpu"))
        state = merge_expert_and_shared_state(expert_state, shared_state)
        filename = os.path.join(new_domain_folder_path, 'checkpoint_last.pt')
        with open(filename, 'wb') as f:
            torch.save(state, filename)


if __name__=='__main__':
    args = add_args()
    main(args.old_folder, args.new_folder, args.subfolder, args.new_subfolder, args.load_from_step, args.domain_id)
