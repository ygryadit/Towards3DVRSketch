import bpy
import sys
import numpy as np
from os.path import join, basename
from mathutils import Vector

context = bpy.context
scene = bpy.context.scene
scene.render.resolution_x = 224
scene.render.resolution_y = 224
# bpy.context.preferences.themes[0].view_3d.space.gradients.high_gradient = (0,0,0)
scene.render.film_transparent = True

def spherical_to_euclidian(elev, azimuth, r):
    x_pos = r * np.cos(elev/180.0*np.pi) * np.cos(azimuth/180.0*np.pi)
    y_pos = r * np.cos(elev/180.0*np.pi) * np.sin(azimuth/180.0*np.pi)
    z_pos = r * np.sin(elev/180.0*np.pi)
    return x_pos, y_pos, z_pos

def render_sketch(model_path, save_dir, nviews = 12):
    model_name = basename(model_path)[:-4]
    bpy.data.cameras['Camera'].type = 'ORTHO'
    bpy.data.cameras['Camera'].ortho_scale = 1.5
    # bpy.data.objects['Cube'].select_set(state=True)
    bpy.ops.object.delete()
    bpy.ops.import_scene.obj(filepath=model_path, filter_glob="*.obj")

    
    
    
    bpy.context.view_layer.objects.active = imported
    bpy.ops.object.convert(target='CURVE', keep_original=False)
    imported.data.bevel_depth = 0.005

    # bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
    local_bbox_center = 0.125 * sum((Vector(b) for b in imported.bound_box), Vector())
    imported.location = local_bbox_center

    imported.rotation_mode = 'XYZ'

    views = np.linspace(0, 2 * np.pi, nviews, endpoint=False)
    print(views)

    # Set target of a camera camera:
    cam = scene.objects['Camera']
    x_pos_, y_pos_, z_pos_ = spherical_to_euclidian(30, 0, 1.0)
    cam.location = (x_pos_, y_pos_, z_pos_)

    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    cam_constraint.target = imported

    # Render views:
    for i in range(nviews):
        # imported.rotation_euler[1] = np.pi
        imported.rotation_euler[2] = views[i]
        # bpy.ops.view3d.camera_to_view_selected()
        context.scene.render.filepath = join(save_dir, model_name + "_" + str(i) + ".png")
        bpy.ops.render.render(write_still=True)

    meshes_to_remove = []
    for ob in bpy.context.selected_objects:
        meshes_to_remove.append(ob.data)
    bpy.ops.object.delete()
    # Remove the meshes from memory too
    for mesh in meshes_to_remove:
        bpy.data.curves.remove(mesh)

    imported = None
    del imported

def spherical_to_euclidian(elev, azimuth, r):
    x_pos = r * np.cos(elev/180.0*np.pi) * np.cos(azimuth/180.0*np.pi)
    y_pos = r * np.cos(elev/180.0*np.pi) * np.sin(azimuth/180.0*np.pi)
    z_pos = r * np.sin(elev/180.0*np.pi)
    return x_pos, y_pos, z_pos

def find_longest_diagonal(imported):
    local_bbox_center = 0.125 * sum((Vector(b) for b in imported.bound_box), Vector())
    ld = 0.0
    for v in imported.bound_box:
        lv = Vector(local_bbox_center) - Vector(v)
        #print(lv.length)
        ld = max(ld, lv.length)
    return ld
    

def move_camera(coord):
    def deg2rad(deg):
        return deg * np.pi / 180.

    r = 1.
    theta, phi = deg2rad(coord[0]), deg2rad(coord[1])
    loc_x = r * np.sin(theta) * np.cos(phi)
    loc_y = r * np.sin(theta) * np.sin(phi)
    loc_z = r * np.cos(theta)

    scene.objects['Camera'].location = (loc_x, loc_y, loc_z)

def render_mesh(model_path, save_dir, nviews = 12):

    cameras = [(60, i) for i in range(0, 360, 30)]

    # model_name = basename(model_path)[:-4]
    model_name = model_path.split('\\')[-2]
    print("model_name: " + model_name)
    
    bpy.data.cameras['Camera'].type = 'ORTHO'
    bpy.data.cameras['Camera'].ortho_scale = 1.5
    # bpy.data.objects['Cube'].select_set(state=True)
    bpy.ops.object.delete()

    bpy.ops.import_scene.obj(filepath=model_path, filter_glob="*.obj")

    print( bpy.context.view_layer.objects.active)

    imported_all = bpy.context.selected_objects[:]
    for obj in imported_all:
        bpy.context.view_layer.objects.active = obj
    
    print(bpy.context.selected_objects)

    print(bpy.context.view_layer.objects.active)
    
    bpy.ops.object.join()
    print(bpy.context.selected_objects)

    imported = bpy.context.selected_objects[0]        
    # imported = bpy.context.selected_objects[0]
    # imported_all = bpy.context.selected_objects[1:]
    
    # go edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)           
    bpy.ops.mesh.faces_shade_smooth()
    bpy.ops.object.editmode_toggle()
    
        # # select al faces
        # bpy.ops.mesh.select_all(action='SELECT')
        # # recalculate outside normals 
        # bpy.ops.mesh.normals_make_consistent(inside=False)
        # # go object mode again
        # bpy.ops.object.editmode_toggle()
        
        
    bpy.context.view_layer.objects.active = imported
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
    maxDimension = 0.56
    ld = find_longest_diagonal(imported)
    #scaleFactor = maxDimension / max(imported.dimensions)
    scaleFactor = maxDimension / ld
    imported.scale = (scaleFactor,scaleFactor,scaleFactor)
    #local_bbox_center = 0.125 * sum((Vector(b) for b in imported.bound_box), Vector())
    #imported.location = local_bbox_center

    imported.rotation_mode = 'XYZ'

    views = np.linspace(0, 2 * np.pi, nviews, endpoint=False)
    print(views)

    # Set target of a camera camera:
    cam = scene.objects['Camera']
    # x_pos_, y_pos_, z_pos_ = spherical_to_euclidian(30, 0, 1.0)
    # cam.location = (x_pos_, y_pos_, z_pos_)

    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    
    # Track to origin:
    origin_name = 'origin'
    origin = bpy.data.objects[origin_name] 
    origin.location = (0, 0, 0)
    cam_constraint.target = origin

    # for i in range(nviews):
        # # imported.rotation_euler[2] = np.pi / 2 + views[i]
        # # imported.rotation_euler[1] = np.pi
        # imported.rotation_euler[2] = views[i]
        # # imported.rotation_euler[0] = np.pi
        # # bpy.ops.view3d.camera_to_view_selected()
        # context.scene.render.filepath = join(save_dir, model_name + "_" + str(i) + ".png")
        
        # bpy.ops.render.render(write_still=True)

    for i, c in enumerate(cameras):
        move_camera(c)
        context.scene.render.filepath = join(save_dir, model_name + "_" + str(i) + ".png")        
        bpy.ops.render.render(write_still=True)

    meshes_to_remove = []
    for ob in bpy.context.selected_objects:
        meshes_to_remove.append(ob.data)
    bpy.ops.object.delete()
    # Remove the meshes from memory too
    for mesh in meshes_to_remove:
        bpy.data.meshes.remove(mesh)

    imported = None
    del imported

if __name__ == '__main__':
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]
    model_path = argv[0]
    export_dir = argv[1]
    print(model_path)
    print(export_dir)
    type = argv[2]
    if type == 'network':
        render_sketch(model_path, export_dir)
    else:
        render_mesh(model_path, export_dir)