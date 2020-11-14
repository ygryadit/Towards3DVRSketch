import numpy as np
import random
from math import pi, cos, sin
from scipy import interpolate
import os
from pyknotid.spacecurves import Knot
from glob import glob
from similaritymeasures import frechet_dist, area_between_two_curves, curve_length_measure
from sklearn.cluster import AgglomerativeClustering
import shutil

def m_trans(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]
                     ])


def m_rot(MAX_ANGLE, F):
    a = random.uniform(-MAX_ANGLE, MAX_ANGLE) * F
    b = random.uniform(-MAX_ANGLE, MAX_ANGLE) * F
    c = random.uniform(-MAX_ANGLE, MAX_ANGLE) * F
    rot_z = np.array([[cos(a), -sin(a), 0, 0],
                       [sin(a), cos(a), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]
                       ])
    rot_y = np.array([[cos(b), 0, -sin(b), 0],
                       [0, 1, 0, 0],
                       [sin(b), 0, cos(b), 0],
                       [0, 0, 0, 1]
                       ])
    rot_x = np.array([[1, 0, 0, 0],
                       [0, cos(c), sin(c), 0],
                       [0, -sin(c), cos(c), 0],
                       [0, 0, 0, 1]
                       ])
    return rot_z.dot(rot_y).dot(rot_x)


def m_scale(sx, sy, sz):
    return np.matrix([[sx, 0, 0, 0],
                      [0, sy, 0, 0],
                      [0, 0, sz, 0],
                      [0, 0, 0, 1]])


def random_scale(minimum, maximum, F, par):
    minimum = 1 + (minimum - 1) * F
    maximum = 1 + (maximum - 1) * F

    if par:
        r1 = random.uniform(minimum, maximum)
        r2 = r1
        r3 = r1
    else:
        r1 = random.uniform(minimum, maximum)
        r2 = random.uniform(minimum, maximum)
        r3 = random.uniform(minimum, maximum)

    return m_scale(r1, r2, r3)


def random_translate(max_dist, F):
    theta = random.uniform(-pi, pi)
    norm = random.uniform(0, max_dist) * F
    tx, ty = norm * cos(theta), norm * sin(theta)
    tz = random.uniform(0, max_dist) * F

    return m_trans(tx, ty, tz)


def getRandomTransform(point_list, F, max_scale):
    MAX_ANGLE = pi * 10 / 180
    MIN_SCALE = 0.9
    MAX_SCALE = 1.1
    MAX_TRANSLATE = max_scale * 0.1
    PRESERVE_RATIO = 0

    center = np.sum(point_list, axis=0) / len(point_list)
    T_ori = m_trans(-center[0], -center[1], -center[2])

    R = m_rot(MAX_ANGLE, F)
    S = random_scale(MIN_SCALE, MAX_SCALE, F, PRESERVE_RATIO)
    T_inv = m_trans(center[0], center[1], center[2])
    T = random_translate(MAX_TRANSLATE, F)

    M = T.dot(T_inv).dot(S).dot(R).dot(T_ori)

    return M

def read_obj(model_path):
    objFile = open(model_path, 'r')
    vertexList = []
    lineList = []
    chainList = []
    for line in objFile:
        split = line.split()
        # if blank line, skip
        if not len(split):
            continue
        if split[0] == "v":
            vertexList.append([float(split[1]), float(split[2]), float(split[3])])
        elif split[0] == "l":
            lineList.append(split[1:])
        elif split[0] == "#":
            chainList.append(split[2:])
    objFile.close()
    return vertexList, lineList, chainList

import codecs

def save_string_list(file_path, v, l, is_utf8=False):
    if is_utf8:
        f = codecs.open(file_path, 'w', 'utf-8')
    else:
        f = open(file_path, 'w')
    for item in v:
        f.write('v %.6f %.6f %.6f \n' % (item[0], item[1], item[2]))
    for item in l:
        f.write('l %s %s \n' % (item[0], item[1]))
    if len(l) >= 1:
        f.write('l %s %s \n' % (l[-1][0], l[-1][1]))
    f.close()

def get_points(chain, lineList, vertexList):
    #     chain = chainList[10]
    line = chain[0]
    l = lineList[int(line) - 1]
    id0, id1 = int(l[0]) - 1, int(l[1]) - 1
    vertices = [vertexList[id0], vertexList[id1]]
    for line in chain[1:]:
        l = lineList[int(line) - 1]
        id1 = int(l[1]) - 1
        vertices.append(vertexList[id1])
    return vertices

def extend_stroke(a, b, p):
    v = np.array(b) - np.array(a)
    if np.linalg.norm(v) != 0:
        v = v / np.linalg.norm(v)
    return (b + random.uniform(0, p) * v).tolist()

def select_split(vertices, split_num):
    """

    :param vertices: list of vertices
    :param split_num:
    :return:
    """
    split_strokes = []
    points = np.array(vertices)
    K = Knot(points)
    curvatures = K.curvatures()
    curvatures[np.isnan(curvatures)] = 0
    bias = 1
    valid_curvatures = curvatures[bias:-bias]
    curve_indexes = np.where(valid_curvatures > np.mean(valid_curvatures) * 2)[0]
    if len(curve_indexes) == 0:
        return [vertices]
    # print(curve_indexes)
    curve_indexes = sorted(random.sample(list(curve_indexes), min(split_num, len(curve_indexes))))
    pre_index = 0
    for index in curve_indexes:
        index += bias
        split_strokes.append(vertices[pre_index:index])
        pre_index = index - 1
    #     assert len(vertices[pre_index:index]) > 1
    # assert len(vertices[pre_index:]) > 1
    split_strokes.append(vertices[pre_index:])
    return split_strokes

def sample_vertice(stroke, ratio):
    """
    Sampling the vertices of input stroke
    :param stroke: list of vertices
    :param ratio: sampling ratio, decide the num of vertices to keep
    :return: list of vertices after sampling
    """
    if len(stroke) <= 2:
        return stroke
    num = len(stroke) - 2
    sample_amount = int(num * ratio)
    if sample_amount == 0:
        sample_amount = 1
    sampled_vertices = [stroke[i + 1] for i in sorted(random.sample(range(num), sample_amount))]
    sampled_vertices.insert(0, stroke[0]) # add first vertice of original stroke
    sampled_vertices.append(stroke[-1]) # add last vertice of original stroke
    return sampled_vertices

def interpolate_vertices(vertices, step = 0.001):
    vertices_array = np.array(vertices)
    v0 = vertices_array[0]
    new_vertices = [v0]
    for v1 in vertices_array[1:]:
        v = v1 - v0
        n = int(np.linalg.norm(v) / step)
        grid = np.linspace(0, 1, n)
        for i in grid[1:-1]:
            new_v = v0 + v * i
            new_vertices.append(new_v)
        new_vertices.append(v1)
        v0 = v1
    return new_vertices

def split_stroke(chain_vertex_list, max_scale, interpolate=False, split_num=4, ratio=0.4,
                 over_stroke=False, extend=False, filter=False, limit=5):
    """
    
    :param chainList: 
    :param lineList: 
    :param vertexList: 
    :param max_scale: 
    :param interpolate: 
    :param split_num: 
    :param ratio: 
    :param over_stroke: 
    :param extend: 
    :param filter: whether to filter short strokes
    :param limit: the num limit of vertices to filter strokes
    :return: 
    """

    new_chains = []
    for vertices in chain_vertex_list:
        # vertices = get_points(chain, lineList, vertexList)
        if interpolate:
            # print("len of before interpolate: ", len(vertices))
            vertices = interpolate_vertices(vertices)
            # print("len of vertices after interpolate: ", len(vertices))
        if len(vertices) > 4:
            split_strokes = select_split(vertices, split_num)
            new_chains.extend(split_strokes)
        else:
            new_chains.append(vertices)

    if filter:
        limit = limit * max_scale
        filtered_list = []
        for vertices in new_chains:
            dist = get_length(vertices)
            if dist > limit:
                filtered_list.append(vertices)
        new_chains = filtered_list

    if over_stroke:
        over_list = []
        for chain in chainList:
            vertices = get_points(chain, lineList, vertexList)
            split_strokes = select_split(vertices, split_num)
            over_list.extend(split_strokes)
        new_chains.extend(random.sample(over_list, int(len(new_chains) * 0.1)))

    sampled_list = [sample_vertice(chain, ratio) for chain in new_chains]

    if extend:
        p = max_scale * 0.1
        for i, vertices in enumerate(sampled_list):
            if len(vertices) > 1:
                # print("len of vertices: ", len(vertices))
                a, b = vertices[1], vertices[0]
                sampled_list[i].insert(0, extend_stroke(a, b, p))
                a, b = vertices[-2], vertices[-1]
                sampled_list[i].append(extend_stroke(a, b, p))
    return sampled_list


def get_point_list(chain, vertexList):
    point_list = []
    for index in chain:
        v = vertexList[index - 1]
        point_list.append(v)
    return np.array(point_list)


def coherentDisturb(a, b, p, F):
    """

    :param a:
    :param b:
    :param p: noise
    :param F:
    :return:
    """
    u = np.array(b) - np.array(a)
    dx, dy, dz = random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)

    if u[2] == 0:
        if u[1] != 0:
            v = np.array([dx, -u[0] * dx / u[1].item(0), dz])
        else:
            v = np.array([0, dy, dz])
    else:
        v = np.array([dx, dy, (-u[0] * dx - u[1] * dy) / u[2].item(0)])
    if np.linalg.norm(v) != 0:
        v = v / np.linalg.norm(v) * random.uniform(-p, p) * F
    return m_trans(v[0], v[1], v[2])  # (b + random.uniform(-p, p) * F * v).tolist()


def get_edge_list(vertice_array):
    edge_list = []
    p_count = 0
    for point_list in vertice_array:
        p_count += 1
        for i in range(len(point_list) - 1):
            edge_list.append([p_count, p_count + 1])
            p_count += 1
    return edge_list

def disturb_obj(vertice_list, max_scale, F_range=1.0, noise=0.01, coherent=False):
    """

    :param vertice_list:
    :param max_scale:
    :param F_range:
    :param noise:
    :param coherent:
    :return:
    """
    disturb_array = []
    for point_list in vertice_list:
        sum_distance = get_length(point_list)

        local_noise = noise * sum_distance # /0.005
        # print(sum_distance, noise)
        stroke_list = []
        F = random.uniform(0.0, F_range)
#         point_list = get_point_list(chain, vertexList)
        M_local = getRandomTransform(point_list, F, max_scale)
        # add the first vertice of the stroke
        pre_v = point_list[0]
        P = np.transpose(np.array([pre_v[0], pre_v[1], pre_v[2], 1]))
        T_noise = random_translate(local_noise, F)
        p_rot = T_noise.dot(M_local).dot(P)
        stroke_list.append([p_rot.item(0), p_rot.item(1), p_rot.item(2)])

        # disturb the second vertice of each line and add to the list
        for i, v in enumerate(point_list[1:]):
            if coherent:
                T_noise = coherentDisturb(pre_v, v, local_noise, F)
            else:
                T_noise = random_translate(local_noise, F)
            P = np.transpose(np.array([v[0], v[1], v[2], 1]))
            p_rot = T_noise.dot(M_local).dot(P)
            stroke_list.append([p_rot.item(0), p_rot.item(1), p_rot.item(2)])
            pre_v = v
        disturb_array.append(stroke_list)
    return disturb_array

def get_length(stroke):
    p1 = stroke[0]
    sum_distance = 0
    for p2 in stroke[1:]:
        squared_dist = np.sum((p1-p2)**2, axis=0)
#         print(np.sqrt(squared_dist))
        sum_distance += np.sqrt(squared_dist)
        p1 = p2
    return sum_distance

def smooth_stroke(new_chains, s=2, step=0.005):
    """

    :param new_chains:
    :param s:
    :param step:
    :return:
    """
    interpolated_chains = []
    for chain in new_chains:
        chain_array = np.array(chain)
        sum_distance = get_length(chain_array)
        num_true_pts = int(sum_distance / step)
        if num_true_pts > 2:
            # print(chain_array, s * len(chain_array))
            tck, u = interpolate.splprep([chain_array[:, 0], chain_array[:, 1], chain_array[:, 2]], s=s * len(chain_array))
            # print("num_true_pts: {}".format(num_true_pts))
            u_fine = np.linspace(0, 1, num_true_pts)
            x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
            new_array = np.array([point for point in zip(x_fine, y_fine, z_fine)])
            interpolated_chains.append(new_array)
        else:
            interpolated_chains.append(chain_array)
    return interpolated_chains

def lines_dist(chain1, chain2):
    mean1 = np.mean(chain1, axis=0)
    mean2 = np.mean(chain2, axis=0)
    v = np.linalg.norm(mean1 - mean2)
    return v

def get_clustered_chains(new_chains):
    sim_array = np.zeros([len(new_chains), len(new_chains)])
    dist_array = np.zeros([len(new_chains), len(new_chains)])
    for i in range(len(new_chains)):
        for j in range(i + 1, len(new_chains)):
            v = np.array(new_chains[j][0]) - np.array(new_chains[i][0])
            x_trans = np.array(new_chains[i]) + v

            sim_array[i][j] = frechet_dist(x_trans, new_chains[j])
            # sim_array[i][j] = area_between_two_curves(new_chains[i], new_chains[j])
            # sim_array[i][j] = curve_length_measure(new_chains[i], new_chains[j])
            dist_array[i][j] = lines_dist(new_chains[i], new_chains[j])
    sim_array = sim_array + sim_array.T
    dist_array = dist_array + dist_array.T
    return sim_array, dist_array

def get_final_chain(n_clusters, sim_array, dist_array, sample_num = 2):
    if n_clusters > len(sim_array):
        n_clusters = len(sim_array)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed',
                                         linkage='average').fit_predict(sim_array)
    final_chain = []
    for cluster_id in range(n_clusters):
        class_list = np.where(clustering == cluster_id)[0].tolist()
        if len(class_list) > 2:
            count = 0
            while count < sample_num and len(class_list) >= 2:
                subset = dist_array[class_list, :][:, class_list]
                max_index = np.unravel_index(subset.argmax(), subset.shape)
                sample = [class_list[max_index[0]], class_list[max_index[1]]]

                # extra_sample = np.random.choice(class_list, max(sample_num, 1), replace=False)
                # print(class_list, max_index[0], max_index[1])
                remove_items = []
                nearest = np.where(subset[max_index[0], :] < subset[max_index[0], :].mean())[0]
                remove_items.extend([class_list[item] for item in nearest])
                nearest = np.where(subset[max_index[1], :] < subset[max_index[1], :].mean())[0]
                remove_items.extend([class_list[item] for item in nearest])

                remove_items = set(remove_items)
                # print(subset[max_index[0], :].shape, nearest, class_list)
                # class_list_copy = class_list.copy()
                for item in remove_items:
                    class_list.remove(item)

                final_chain.extend(sample)
                count += 2
        else:
            final_chain.extend(class_list)
    return final_chain
# def main(model_path, save_dir, thetas = [0.5], filter=True):
def main(work_info):
    model_path, save_dir, thetas = work_info
    filter = True
    # save_file_name = os.path.basename(model_path).split('_opt')[0] + '_sketch_1.0.obj'
    # save_path = os.path.join(save_dir, save_file_name)
    # if os.path.exists(save_path):
    #     return None
    split_num = 5  # int(theta * 10)
    ratio = 0.01 #0.05 #0.01 #/ theta # should be controlled in [0,1]
    limit = 0.2 #int(ratio * 100)
    smooth = 1 #0.1 / theta # theta ** 2  # * 0.1

    cluster_ratio = {
        0.0:1.0,
        0.25:0.8,
        0.5:0.6,
        0.75:0.4,
        1.0:0.2
    }
    vertexList, lineList, chainList = read_obj(model_path)
    if len(vertexList) == 0:
        print("Failed:" + model_path)
        folder_base = r'S:\Research\VR_Sketch\shapenet_chair\network\success'
        file_from =  os.path.join(folder_base, os.path.basename(model_path))
        for theta in thetas:
            save_file_name = os.path.basename(model_path).split('.')[0] + '_sketch_{}.obj'.format(theta)
            save_path = os.path.join(save_dir, save_file_name)
            print("file_from: " + file_from)
            print("save_path: " + save_path)
            #shutil.copyfile(model_path, save_path)        
        return

    max_scale = np.max(np.array(vertexList))

    all_chains = []
    for chain in chainList:
        vertices = get_points(chain, lineList, vertexList)
        all_chains.append(vertices)
    sim_array, dist_array = get_clustered_chains(all_chains)
    try:
        for theta in thetas:
            noise = 0.1 * theta
            F_range = theta * 1.5

            if theta > 0:
                n_clusters = max(int(len(all_chains) * cluster_ratio[theta] * 0.5), 10)
                final_chain = get_final_chain(n_clusters, sim_array, dist_array, sample_num=4)
                # print(final_chain, len(all_chains))
                chain_vertex_list = [all_chains[item] for item in final_chain]
            else:
                chain_vertex_list = all_chains
            new_chains = split_stroke(chain_vertex_list, max_scale, interpolate=True, split_num=split_num,
                                      ratio=ratio, over_stroke=False, extend=True, filter=filter, limit=limit)

            disturb_array = disturb_obj(new_chains, max_scale, F_range=F_range, noise=noise, coherent=True)
            interpolated_array = smooth_stroke(disturb_array, s=smooth)

            vertice_list = [item for stroke in interpolated_array for item in stroke]
            edge_list = get_edge_list(interpolated_array)

            save_file_name = os.path.basename(model_path).split('.')[0] + '_sketch_{}.obj'.format(theta)
            save_path = os.path.join(save_dir, save_file_name)
            print(save_path)

            save_string_list(save_path, vertice_list, edge_list)
    except:
        print("Failed:" + model_path)
        save_file_name = os.path.basename(model_path).split('.')[0] + '_sketch_{}.obj'.format(theta)
        save_path = os.path.join(save_dir, save_file_name)
        shutil.copyfile(model_path, save_path)
        print(save_path)
        return
        
if __name__ == '__main__':

    argv = sys.argv
    argv = argv[argv.index("--") + 1:]
    
    root_dir = argv[0]
    save_dir = argv[1]    
      
    model_paths = glob(os.path.join(root_dir, '*.obj'))
        
    model_paths_not_processed = []
    for model_path in model_paths:
        save_file_name = os.path.basename(model_path).split('.')[0] + '_sketch_{}.obj'.format(1.0)
        save_path = os.path.join(save_dir, save_file_name)
        if not os.path.exists(save_path):
            model_paths_not_processed.append(model_path)
    
  

    thetas = [0.0, 0.25, 0.5, 0.75, 1.0]
    

    work_info = [(path, save_dir, thetas) for path in model_paths_not_processed] #filtered_models]
    from multiprocessing import Pool

    with Pool(4) as p:
        
        p.map(main, work_info)
        
        
            
