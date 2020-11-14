import os
from glob import glob
import subprocess
from multiprocessing import Pool
import shutil



def aggregate(work_info):
    input_folder, path, save_dir = work_info
    # print(path)
    save_path = path[:-4] + '_aggredated.obj'
    model_name = os.path.basename(path)
    target_path = os.path.join(save_dir, model_name[:-4] + '_aggredated.obj')
    if os.path.exists(target_path):
        return
    input_folder = os.path.join(input_folder, '') 
    print(input_folder)
    print(model_name)
    cmd = r'merge_lines_bin.exe -in %s -model_name %s' % (input_folder, model_name)
    p = subprocess.Popen(cmd.split(' '))
    try:
        p.wait(1000)
    except subprocess.TimeoutExpired:
        p.kill()
    print(save_path)
    if os.path.exists(save_path):
        shutil.move(save_path, save_dir)

if __name__ == '__main__':

    argv = sys.argv
    argv = argv[argv.index("--") + 1:]
    
    folder_netwroks = argv[0]
    folder_save = argv[1]    
    executable_path = argv[2]
    
    os.chdir(executable_path)

    input_folder = r'\\surrey.ac.uk\Research\vssp_datasets\multiview\3VS\network'
    input_folder_n = r'S:\Research\VR_Sketch\shapenet_chair\network\success'
    
    save_dir = r'S:\Research\VR_Sketch\shapenet_chair\merge_sketch'
      
    model_paths = glob(os.path.join(input_folder, '*.obj'))
        
    work_info = [(input_folder_n, path, save_dir) for path in model_paths]
    
    with Pool(4) as p:
        p.map(aggregate, work_info)
    