import numpy as np
import torch
from torch.utils.data.sampler import Sampler, BatchSampler
from itertools import combinations, permutations
import torch.nn.functional as F

class BalancedBatchSampler(BatchSampler):
    def __init__(self, labels, n_classes, n_samples, seed, n_dataset=None):
        self.labels = labels
        self.labels_set = list(sorted(set(self.labels)))
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0] for label in self.labels_set}

        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])

        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples # sample for sketch/shape
        if n_dataset is not None:
            self.n_dataset = n_dataset
        else:
            self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes
        self.seed = seed#self.n_dataset // self.batch_size

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            np.random.seed(self.seed + int(self.count/self.batch_size)) # assure any batch contain anchor/other samples from the same classes set
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                index = self.used_label_indices_count[class_]
                indices.extend(self.label_to_indices[class_][index:index + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.batch_size

    def __len__(self):
        return self.n_dataset // self.batch_size

def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix

class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError

def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None

def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None

class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, sketch_anchor=False, anchor_index=0, cpu=False):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn
        self.sketch_anchor = sketch_anchor
        self.anchor_index = anchor_index

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            if self.sketch_anchor:
                negative_indices = np.where(np.logical_not(label_mask[self.anchor_index:]))[0] + self.anchor_index

                list_anchor = [i for i in label_indices if i < self.anchor_index]
                list_positive = [i for i in label_indices if i >= self.anchor_index]
                anchor_positives = [[i ,j] for i in list_anchor for j in list_positive]
            else:
                negative_indices = np.where(np.logical_not(label_mask))[0]
                anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)

def HardestNegativeTripletSelector(margin, sketch_anchor, anchor_index, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                              sketch_anchor=sketch_anchor,

                                                                                              anchor_index=anchor_index,
                                                                                              cpu=cpu)

def RandomNegativeTripletSelector(margin,  sketch_anchor, anchor_index, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                negative_selection_fn=random_hard_negative,
                                                                                             sketch_anchor=sketch_anchor,
                                                                                             anchor_index=anchor_index,
                                                                                             cpu=cpu)


def SemihardNegativeTripletSelector(margin, sketch_anchor, anchor_index, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                  negative_selection_fn=lambda x: semihard_negative(x, margin),
                                                                                  sketch_anchor=sketch_anchor,
                                                                                  anchor_index=anchor_index,
                                                                                  cpu=cpu)
if __name__ == "__main__":
    from config import DATASETS
    from dataset.ImageLoader import ImgDataset
    from dataset.MultiViewLoader import MultiviewImgDataset
    from dataset.PointCloudLoader import PointCloudDataLoader
    dataset = 'ModelNet10'
    list_file = DATASETS[dataset]['list_file']
    # train_dataset = ImgDataset(set='train', list_file=list_file, scale_aug=False, rot_aug=False,
    #                                     shuffle=False)
    # train_dataset = ImgDataset(set='valid', list_file=list_file, scale_aug=False, rot_aug=False,
    #                                     shuffle=False)
    # train_dataset = PointCloudDataLoader(list_file, split='train', uniform=False, sketch=False, abstract=0.5)

    sketch_train_dataset = MultiviewImgDataset(set='train', list_file=list_file, scale_aug=False, rot_aug=False, num_views=12,
                                        shuffle=False, shape=False)
    sketch_train_batch_sampler = BalancedBatchSampler(sketch_train_dataset.labels, n_classes=2, n_samples=2)
    sketch_dataloader = torch.utils.data.DataLoader(sketch_train_dataset, batch_sampler=sketch_train_batch_sampler)

    shape_train_dataset = MultiviewImgDataset(set='train', list_file=list_file, scale_aug=False, rot_aug=False, num_views=12,
                                        shuffle=False, sketch=False)
    shape_train_batch_sampler = BalancedBatchSampler(shape_train_dataset.labels, n_classes=2, n_samples=2)
    shape_dataloader = torch.utils.data.DataLoader(shape_train_dataset, batch_sampler=shape_train_batch_sampler)

    # for images, labels in DataLoader:
    for epoch in range(3):
        sketch_train_batch_sampler.seed += sketch_train_batch_sampler.n_dataset // sketch_train_batch_sampler.batch_size
        shape_train_batch_sampler.seed += shape_train_batch_sampler.n_dataset // shape_train_batch_sampler.batch_size

        for i, ((sketches, k_labels), (shapes, p_labels)) in enumerate(zip(sketch_dataloader, shape_dataloader)):
            print(k_labels, p_labels)
            if i > 3:
                print('Epoch {} over'.format(epoch))
                break
