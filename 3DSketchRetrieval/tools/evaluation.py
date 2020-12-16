import numpy as np
# from sklearn.metrics.pairwise import euclidean_distances
import torch
# from rank_metrics import mean_average_precision
# import ipdb

def compute_acc_at_k(dist):
    pair_sort = np.argsort(dist)
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
    return [acc_1, acc_5, acc_10]

def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

def average_precision(r):
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k+1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)

def mean_average_precision(rs):
    return np.mean([average_precision(r) for r in rs])

def map_and_auc(label_q, label_d, d):
    rs = convert_rank_gt(label_q, label_d, d)
    trec_precisions = []
    mrecs = []
    mpres = []
    aps = []
    for i, r in enumerate(rs):
        #ipdb.set_trace()
        res = precision_and_recall(rs[i])
        trec_precisions.append(res[0])
        mrecs.append(res[1])
        mpres.append(res[2])
        aps.append(res[3])

    trec_precisions = np.stack(trec_precisions)
    mrecs = np.stack(mrecs)
    mpres = np.stack(mpres)
    aps = np.stack(aps)
    AUC = np.mean(aps)
    mAP = np.mean(trec_precisions)
    return AUC, mAP

def compute_map(label_q, label_d, d):
    rs = convert_rank_gt(label_q, label_d, d)
    return mean_average_precision(rs)

def convert_rank_gt(label_q, label_d, d):
    idx = d.argsort(axis=1)
    label_q.resize(label_q.size, 1)
    label_d.resize(1, label_d.size)
    gt = (label_q == label_d)
    rs = [gt[i][idx[i]] for i in range(gt.shape[0])] # rank ground truth
    return rs


def precision_and_recall(r):
    num_gt = np.sum(r)
    trec_precision = np.array([np.mean(r[:i+1]) for i in range(r.size) if r[i]])
    recall = [np.sum(r[:i+1])/num_gt for i in range(r.size)]
    precision = [np.mean(r[:i+1]) for i in range(r.size)]

    # interpolate it
    mrec = np.array([0.] + recall + [1.])
    mpre = np.array([0.] + precision + [0.])

    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])

    i = np.where(mrec[1:] != mrec[:-1])[0]+1
    ap = np.sum((mrec[i]-mrec[i-1]) * mpre[i])
    return trec_precision, mrec, mpre, ap


def plot_pr_cure(mpres, mrecs):
    pr_curve = np.zeros(mpres.shape[0], 10)
    for r in range(mpres.shape[0]):
        this_mprec = mpres[r]
        for c in range(10):
            pr_curve[r, c] = np.max(this_mprec[mrecs[r]>(c-1)*0.1])
    return pr_curve

def l2_normalize(features):
    # features: num * ndim
    features_c = features.copy()
    features_c /= np.sqrt((features_c * features_c).sum(axis=1))[:, None]
    return features_c

# def compute_distance(x, y, l2=True):
#     if l2:
#         x = l2_normalize(x)
#         y = l2_normalize(y)
#     return euclidean_distances(x, y)

def compute_distance(a, b, l2=True):
    if l2:
        a = l2_normalize(a)
        b = l2_normalize(b)

    """cdist (squared euclidean) with pytorch"""
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    a_norm = (a**2).sum(1).view(-1, 1)
    b_t = b.permute(1, 0).contiguous()
    b_norm = (b**2).sum(1).view(1, -1)
    dist = a_norm + b_norm - 2.0 * torch.matmul(a, b_t)
    dist[dist != dist] = 0
    return torch.clamp(dist, 0.0, np.inf).numpy()