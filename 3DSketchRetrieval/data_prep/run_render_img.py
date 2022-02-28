import os, sys
from glob import glob
from multiprocessing import Pool

#TODO: Replace with your blender path
blender_path = r'C:\Program Files\Blender Foundation\Blender 2.82' # replace this with your Blender path

def render_Phong(work_info):
    model_path, save_dir, data_type = work_info
    # TODO: may need to replace with absolute path of render_meshes_shaded_black_bg.blend and render_img.py
    os.system(r'blender 3DSketchRetrieval/data_prep/blender/render_meshes_shaded_black_bg.blend --background --python 3DSketchRetrieval/data_prep/render_img.py -- %s %s %s' % (model_path, save_dir, data_type))

def render_depth(work_info):
    model_path, save_dir, data_type = work_info
    # TODO: may need to replace with absolute path of render_depth.blend and render_img.py
    os.system(r'blender 3DSketchRetrieval/data_prep/blender/render_depth.blend --background --python 3DSketchRetrieval/data_prep/render_img.py -- %s %s %s' % (model_path, save_dir, data_type))
    
if __name__ == '__main__':
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]
    obj_dir = argv[0]
    save_dir = argv[1]
    data_type = argv[2] #'sketch' or 'shape'
    render_type = argv[3] #'Phong' or 'depth'

    model_files = glob(os.path.join(obj_dir, '*.obj'))

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    os.chdir(blender_path)
        
    work_info = [(path, save_dir, data_type) for path in model_files]
    if render_type == 'Phong':
        with Pool(2) as p:
            p.map(render_Phong, work_info)
    elif render_type == 'depth':
        with Pool(2) as p:
            p.map(render_depth, work_info)
    else:
        NotImplementedError
