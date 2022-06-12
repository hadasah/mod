import shutil
import mod_checkpoint_utils
import os

CHECKPOINTS_TOP_FOLDER = '/checkpoint/margaretli/mod/'
NEW_MODEL_TOP_FOLDER = '/checkpoint/margaretli/mod_old/'

FOLDERS = mod_checkpoint_utils.find_folders(CHECKPOINTS_TOP_FOLDER, re_string='')

for subfolder in FOLDERS:
    old_folder = os.path.join(CHECKPOINTS_TOP_FOLDER, subfolder)
    new_domain_folder_path = os.path.join(NEW_MODEL_TOP_FOLDER, subfolder)
    src_filename = os.path.join(old_folder, 'checkpoint_last.pt')
    filename = os.path.join(new_domain_folder_path, 'checkpoint_last.pt')
    os.makedirs(new_domain_folder_path, exist_ok=True)
    if not os.path.exists(filename):
        shutil.copyfile(src_filename, filename)