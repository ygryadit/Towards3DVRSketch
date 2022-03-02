from Dataset_Loader import get_test_dataloader
import os, json
import importlib
import torch, time
from tools.evaluation import compute_distance, compute_acc_at_k
import numpy as np
import logging
from pathlib import Path
def pt_cdist(a, b):
    """cdist (squared euclidean) with pytorch"""
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    a_norm = (a**2).sum(1).view(-1, 1)
    b_t = b.permute(1, 0).contiguous()
    b_norm = (b**2).sum(1).view(1, -1)
    dist = a_norm + b_norm - 2.0 * torch.matmul(a, b_t)
    dist[dist != dist] = 0
    return torch.clamp(dist, 0.0, np.inf).numpy()

def log_string(logger, str):
    logger.info(str)
    print(str)

def calc_map(logger, result, labs_query, labs_db, ignorefirst=False):
    aps = []
    for i, db_inds in enumerate(result):
        lab_q = labs_query[i]
        lab_db = labs_db[db_inds]
        if ignorefirst:
            lab_db = lab_db[1:]

        total_ap = 0.
        hit_cnt = 0.
        precisions = []
        for j, l in enumerate(lab_db):
            if l == lab_q:
                hit_cnt += 1.
                total_ap += (hit_cnt / (j + 1))
                precisions.append(hit_cnt / (j + 1))
        total_ap /= hit_cnt
        aps.append(total_ap)

        # print total_ap
        # log_string(logger, '%d class %d, first match: %f' % (i, lab_q, 1 / precisions[0]))

    # mean ap
    m_ap = np.mean(aps)
    log_string(logger,'map:%f' % m_ap)

    # per class
    classes = sorted(list(set(labs_query)))
    classes_aps = [[ap for (ap, clz) in zip(aps, labs_query) if clz == c] for c in classes]
    classes_aps = [np.mean(ap) for ap in classes_aps]
    for ap, clz in zip(classes_aps, classes):
        log_string(logger, 'class%d\t%f' % (clz, ap))

    return m_ap

def l2_normalize(features):
    # features: num * ndim
    features_c = features.copy()
    features_c /= np.sqrt((features_c * features_c).sum(axis=1))[:, None]
    return features_c

def boolize_result(result, labs_query, labs_db):
    labs_db = np.array(labs_db)
    result2 = np.array(result)
    for i, (y1, res) in enumerate(zip(labs_query, result)):
        for j, y2 in enumerate(labs_db[res]):
            result2[i, j] = int(y1 == y2)
    return result2

def dcg(x):
    n = len(x)
    logs = np.log2(range(2, n + 2))  # 2, 3, 4, ..
    ws = (1.0 / logs)  # 1, 1/log(3), ...
    gains = ws * x
    return np.sum(gains)

def idcg(x):
    return dcg(sorted(x, reverse=True))


def ndcg_single(x):
    i = idcg(x)
    if i == 0.0:
        return 0.0
    return dcg(x) / i

def ndcg(logger, result):
    ndcgs = []
    for r in result:
        x = ndcg_single(r)
        ndcgs.append(x)
    ndcg_mean = np.mean(ndcgs)
    log_string(logger, 'NDCG:%f' % ndcg_mean)

def nearest_neighbor(logger, result):
    hit = np.count_nonzero(result[:, 0])
    nn = hit / float(result.shape[0])
    log_string(logger, 'NN:%f' % nn)
    return nn

def validate(sketch_dataloader, shape_dataloader, model, save=False, save_dir='', reconstruct=False):
    sketch_features = []
    shape_features = []
    model = model.eval()
    start_time = time.time()
    recon_list = []

    with torch.no_grad():
        for i, data in enumerate(sketch_dataloader):
            sketch_points = data[0].cuda()
            sketch_points = sketch_points.transpose(2, 1)
            B = sketch_points.shape[0]
            if reconstruct:
                _, sketch_z, recon = model(sketch_points, recon_index=[0, B])
                recon_list.append(recon.transpose(2, 1).data.cpu())

            else:
                _, sketch_z = model(sketch_points, recon_index=[0, B])

            sketch_features.append(sketch_z.data.cpu())
        for i, data in enumerate(shape_dataloader):
            shape_points = data[0].cuda()
            shape_points = shape_points.transpose(2, 1)
            shape_z = model(shape_points, train=False)
            shape_features.append(shape_z.data.cpu())

    shape_features = torch.cat(shape_features, 0).numpy()
    sketch_features = torch.cat(sketch_features, 0).numpy()

    shape_features = l2_normalize(shape_features)
    sketch_features = l2_normalize(sketch_features)

    return shape_features, sketch_features

def calc_acc(logger, pair_sort):
    # pair_sort = np.argsort(dist)
    count_1 = 0
    count_5 = 0
    count_10 = 0
    query_num = pair_sort.shape[0]
    for idx1 in range(0, query_num):
        if idx1 in pair_sort[idx1, 0:1]:
            count_1 = count_1 + 1
        if idx1 in pair_sort[idx1, 0:5]:
            count_5 = count_5 + 1
        if idx1 in pair_sort[idx1, 0:10]:
            count_10 = count_10 + 1
    acc_1 = count_1 / float(query_num)
    acc_5 = count_5 / float(query_num)
    acc_10 = count_10 / float(query_num)
    log_string(logger, 'acc_1:%f' % acc_1)
    log_string(logger, 'acc_5:%f' % acc_5)
    log_string(logger, 'acc_10:%f' % acc_10)

def main(args):
    experiment_dir = Path(args.model_path)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    config_f = os.path.join(log_dir, 'config.json')
    with open(config_f, 'r') as f:
        train_args = json.load(f)
        print(train_args)

    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    from dataset.PointCloudLoader import PointCloudDataLoader
    test_shape_dataset = PointCloudDataLoader(args, list_file=args.list_file, npoint=args.num_point, uniform=args.uniform,
                                              split='test', data_type='shape')
    test_sketch_dataset = PointCloudDataLoader(args, list_file=args.list_file, npoint=args.num_point, uniform=args.uniform,
                                               split='test', data_type='sketch', abstract=args.abstract,
                                               random_sample=args.random_sample)
    test_shape_loader = torch.utils.data.DataLoader(test_shape_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_sketch_loader = torch.utils.data.DataLoader(test_sketch_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    shape_label = np.array(test_shape_dataset.labels)
    sketch_label = np.array(test_sketch_dataset.labels)
    # print(sketch_label)

    model_name = 'pointnet2_cls_msg'  # os.listdir(model_path + '/logs')[1].split('.')[0]
    model = importlib.import_module('models.' + model_name)

    num_class = 10
    classifier = model.get_model(num_class, num_points=train_args['num_point'], decoder=train_args['reconstruct']).cuda().eval()

    model_path = os.path.join(checkpoints_dir, 'best_model.pth')
    assert os.path.exists(model_path), "Model path doesn't exist!"

    checkpoint = torch.load(model_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    shape_features, sketch_features = validate(test_sketch_loader, test_shape_loader, classifier, save=True, save_dir=model_path, reconstruct=args.reconstruct)

    print('start cdist, feature dim: %d, query n: %d, db n: %d' % (sketch_features.shape[1], sketch_features.shape[0], shape_features.shape[0]))
    dist = pt_cdist(sketch_features, shape_features)
    print('done cdist')

    print('start argsort')
    result = np.argsort(dist, axis=1)
    print('done argsort')

    calc_map(logger, result, sketch_label, shape_label, ignorefirst=False)

    result_01 = boolize_result(result, sketch_label, shape_label)
    ndcg(logger, result_01)
    nearest_neighbor(logger, result_01)

    # calc_acc(logger, result)


if __name__ == '__main__':
    from args import get_args
    args = get_args()
    experiment_dir = main(args)