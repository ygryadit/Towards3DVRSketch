import numpy as np
import os
import warnings
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint, fix=False):
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
    if fix:
        np.random.seed(0)
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


class PointCloudDataLoader(Dataset):
    def __init__(self, args, list_file='', npoint=1024, split='train', uniform=False, cache_size=15000, data_type='', abstract=0.5, target='', random_sample=False):
        POINT_DIR = args.data_dir
        self.npoints = npoint
        self.uniform = uniform
        self.list_file = list_file
        self.labels = []
        self.split = split
        self.name_list = [line.rstrip() for line in open(self.list_file.format(split))]
        self.datapath = []
        self.target = target
        abstract_set = [0.0, 0.25, 0.5, 0.75, 1.0]
        if isinstance(abstract, list) and len(abstract) == 1:
            abstract = abstract[0]
        np.random.seed(0)

        if data_type=='shape':
            for line in self.name_list:
                model_name, class_id = line.split(' ')
                shape_path = os.path.join(POINT_DIR, 'shape', model_name + '_opt.txt')
                self.datapath.append(shape_path)
                self.labels.append(int(class_id))

        elif data_type=='sketch':
            if isinstance(abstract, list):
                for line in self.name_list:
                    model_name, class_id = line.split(' ')
                    sketch_paths = [os.path.join(POINT_DIR, 'sketch', model_name + '_sketch_{}.txt'.format(ab)) for ab in abstract]
                    self.datapath.extend(sketch_paths)
                    self.labels.extend([int(class_id)] * len(abstract))
            else:
                for line in self.name_list:
                    model_name, class_id = line.split(' ')
                    if random_sample:
                        abstract = np.random.choice(abstract_set, 1, replace=False)[0]

                    sketch_path = os.path.join(POINT_DIR, 'sketch',  model_name + '_sketch_{}.txt'.format(abstract))
                    self.datapath.append(sketch_path)
                    self.labels.append(int(class_id))
        elif data_type=='network':
            for line in self.name_list:
                model_name, class_id = line.split(' ')
                network_path = os.path.join(POINT_DIR, 'network', model_name + '_opt_quad_network_20_aggredated.txt')
                self.datapath.append(network_path)
                self.labels.append(int(class_id))

        print('The size of %s data is %d' % (split, len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

        if self.target == 'network':
            self.target_datapath = []
            for line in self.name_list:
                model_name, class_id = line.split(' ')
                sketch_path = os.path.join(POINT_DIR, 'network', model_name + '_opt_quad_network_20_aggredated.txt')
                self.target_datapath.append(sketch_path)
            if isinstance(abstract, list):
                self.target_datapath = self.target_datapath * len(abstract)

        elif self.target == 'shape':
            self.target_datapath = []
            for line in self.name_list:
                model_name, class_id = line.split(' ')
                shape_path = os.path.join(POINT_DIR, 'shape', model_name + '_opt.txt')
                self.target_datapath.append(shape_path)
            if isinstance(abstract, list):
                self.target_datapath = self.target_datapath * len(abstract)

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        fix = self.split == 'test'

        if self.target == '':
            if index in self.cache:
                point_set, cls = self.cache[index]
            else:
                file_path = self.datapath[index]
                cls = self.labels[index]
                point_set = np.loadtxt(file_path, delimiter=',').astype(np.float32)
                if self.uniform:
                    point_set = farthest_point_sample(point_set, self.npoints, fix=fix)
                else:
                    point_set = point_set[0:self.npoints, :]

                point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

                if len(self.cache) < self.cache_size:
                    self.cache[index] = (point_set, cls)

            return point_set, cls
        else:
            if index in self.cache:
                point_set, target_point_set, cls = self.cache[index]
            else:
                file_paths = [self.datapath[index], self.target_datapath[index]]
                cls = self.labels[index]
                point_sets = [np.loadtxt(file_path, delimiter=',').astype(np.float32) for file_path in file_paths]

                if self.uniform:
                    point_sets = [farthest_point_sample(point_set, self.npoints, fix=fix) for point_set in point_sets]
                else:
                    point_sets = [point_set[0:self.npoints, :] for point_set in point_sets]

                for point_set in point_sets:
                    point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

                point_set, target_point_set = point_sets
                if len(self.cache) < self.cache_size:
                    self.cache[index] = (point_set, target_point_set, cls)

            return point_set, target_point_set, cls


if __name__ == '__main__':
    import torch
    from config import DATASETS
    list_file = DATASETS['ModelNet10']['list_file']
    # train_dataset = PointCloudDataLoader(list_file, split='train', uniform=False, shape=False, abstract=[0.5, 0.75])
    #
    # data = PointCloudDataLoader(list_file, split='valid', uniform=False, shape=False, abstract=0.5)
    #
    # data = PointCloudDataLoader(list_file, split='test', uniform=False, shape=False, abstract=0.5)
    abstract = [0.5, 0.75]
    num_point = 1024
    n_classes = 8
    n_samples = 1
    test_shape_dataset = PointCloudDataLoader(list_file=list_file, npoint=num_point,
                                              uniform=False,
                                              split='train', sketch=False, abstract=abstract)
    test_sketch_dataset = PointCloudDataLoader(list_file=list_file, npoint=num_point,
                                               uniform=False,
                                               split='train', shape=False, abstract=abstract)

    from dataset.TripletSampler import BalancedBatchSampler

    test_shape_sampler = BalancedBatchSampler(test_shape_dataset.labels, n_classes=n_classes, n_samples=n_samples, seed=0, n_dataset=len(test_sketch_dataset.labels))
    test_shape_loader = torch.utils.data.DataLoader(test_shape_dataset, batch_sampler=test_shape_sampler, num_workers=4)
    test_sketch_sampler = BalancedBatchSampler(test_sketch_dataset.labels, n_classes=n_classes, n_samples=n_samples, seed=0)
    test_sketch_loader = torch.utils.data.DataLoader(test_sketch_dataset, batch_sampler=test_sketch_sampler, num_workers=4)

    # DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    i = 0
    for i, ((sketches, k_labels), (shapes, p_labels)) in enumerate(zip(test_sketch_loader, test_shape_loader)):
        # print(sketches.shape)
        print(k_labels, p_labels)
        i += 1
        if i > 50:
            break
