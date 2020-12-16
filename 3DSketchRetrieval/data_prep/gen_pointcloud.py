import point_cloud_utils as pcu
from glob import glob
import numpy as np
import os, sys

point_num = 10000

def interpolate(v0, v1, step):
    dv = v0 - v1
    length = np.linalg.norm(dv)
    length_list = np.arange(0, length, step)
    point_list = [v0]
    for l_i in length_list:
        point_list.append(v1 + dv/length * l_i)
    return point_list

def read_obj(model_path):
    objFile = open(model_path, 'r')
    vertexList = []
    lineList = []
    for line in objFile:
        split = line.split()
        # if blank line, skip
        if not len(split):
            continue
        if split[0] == "v":
            vertexList.append([float(split[1]), float(split[2]), float(split[3])])
        elif split[0] == "l":
            lineList.append(split[1:])
    objFile.close()
    return vertexList, lineList

def sample_pointcloud_edge(model_path):
    vertexList, lineList = read_obj(model_path)
    if len(vertexList) < 1 or len(lineList) < 2:
        return None
    sum_length = 0
    for edge in lineList:
        v0 = np.array(vertexList[int(edge[0])-1])
        v1 = np.array(vertexList[int(edge[1])-1])
        sum_length += np.linalg.norm(v0 - v1)
    step = sum_length / point_num

    point_list = []
    for edge in lineList:
        v0 = np.array(vertexList[int(edge[0])-1])
        v1 = np.array(vertexList[int(edge[1])-1])
        point_list.extend(interpolate(v0, v1, step))
    sample_index = np.random.choice(len(point_list), point_num, replace=False)
    new_point_list = np.array(point_list)[sample_index]
    return new_point_list

def sample_pointcloud_mesh(obj_path):
    off_v, off_f, off_n = pcu.read_obj(obj_path)
    if off_n.shape[0] != off_v.shape[0]:
        off_n = np.array([])
    v_dense, n_dense = pcu.sample_mesh_random(off_v, off_f, off_n, num_samples=point_num)
    return v_dense

def run_modelnet(work_info):
    model_file, save_dir, data_type = work_info
    model_name = os.path.basename(model_file)[:-4]
    save_path = os.path.join(save_dir, model_name + '.txt')
    # if not os.path.exists(save_path):
    print(model_file)
    if data_type == 'edge':
        point_list = sample_pointcloud_edge(model_file)
    else:
        point_list = sample_pointcloud_mesh(model_file)
    if point_list is None:
        print("Something wrong")
    else:
        np.savetxt(save_path, point_list, delimiter=',', fmt='%1.6f')

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def run_shapenet(work_info):
    model_file, save_dir, data_type = work_info
    model_name = os.path.basename(model_file)[:-4]
    save_path = os.path.join(save_dir, model_name + '.txt')
    if not os.path.exists(save_path) and os.path.exists(model_file):
        # print(model_file)
        if data_type == 'sketch':
            point_list = sample_pointcloud_edge(model_file)
        else:
            point_list = sample_pointcloud_mesh(model_file)
        if point_list is None:
            print("Something wrong:", model_file)
        else:
            np.savetxt(save_path, point_list, delimiter=',', fmt='%1.6f')

            # point_list = np.array(point_list, dtype='float32')
            # np.save(save_path, point_list)

if __name__ == '__main__':
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]
    obj_dir = argv[0]
    save_dir = argv[1]
    data_type = argv[2]
    model_files = glob(os.path.join(obj_dir, '*.obj'))

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    work_info = [(path, save_dir, data_type) for path in model_files] #filtered_models]

    from multiprocessing import Pool

    with Pool(16) as p:
        p.map(run_shapenet, work_info)



