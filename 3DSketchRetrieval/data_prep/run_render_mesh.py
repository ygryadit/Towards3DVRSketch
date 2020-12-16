import os
from glob import glob
from multiprocessing import Pool

blender_path = r'C:\Program Files\Blender Foundation\Blender 2.82' # replace this with your Blender path


def sketch_depth(work_info):
    model_path, save_dir = work_info
    os.system(r'blender S:\Research\VR_Sketch\code\render_meshes_shaded_black_bg.blend --background --python S:\Research\VR_Sketch\code\render_img.py -- %s %s %s' % (model_path, save_dir, 'mesh'))
    #os.system(r'blender S:\Research\VR_Sketch\code\BlenderPhong-master\phong.blend --background --python S:\Research\VR_Sketch\code\BlenderPhong-master\phong.py -- %s %s' % (model_path, save_dir))
    
def run_sketch(root_dir, sketchs_paths):
    # blender_path = r'D:\Blender Foundation\Blender 2.79'
    os.chdir(blender_path)

    #Select the models matching the sketches:
    models = []
    for sketch_name in sketchs_paths:
        models.append(os.path.join(root_dir, os.path.basename(sketch_name).split('_')[1], 'model.obj' ))
    
    
    # Generate views:
    save_dir = r'S:\Research\VR_Sketch\shape_multi_view_w'
    
    work_info = [(path, save_dir) for path in models[:]]
    with Pool(2) as p:
        p.map(sketch_depth, work_info)

if __name__ == '__main__':
    root_dir = r'S:\Research\VR_Sketch\03001627'
    sketchs_paths = glob(r'S:\Research\VR_Sketch\chair_1005_obj\*.obj')

    run_sketch()