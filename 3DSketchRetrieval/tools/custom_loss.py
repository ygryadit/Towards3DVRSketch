import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.nn.functional as F

################################################################
## Triplet related loss
################################################################

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        # prec = (an_distances.data > ap_distances.data).sum().to(dtype=torch.float) / triplets.size(0)  # normalize data by batch size
        prec = (an_distances.data - ap_distances.data).sum().to(dtype=torch.float) / triplets.size(0)
        return losses.mean(), prec #len(triplets)

def pdist(A, squared=False, eps=1e-4):
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min=0)
    return res if squared else (res + eps).sqrt() + eps


class TripletCenterLoss(nn.Module):
    def __init__(self, margin=0, center_embed=10, num_classes=10, l2norm=False):
        super(TripletCenterLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.centers = nn.Parameter(torch.randn(num_classes, center_embed))
        self.l2norm = l2norm

    def forward(self, inputs, targets):
        batch_size = inputs.shape[0]
        targets_expand = targets.view(batch_size, 1).expand(batch_size, inputs.shape[1])
        centers_batch = self.centers.gather(0, targets_expand)  # centers batch

        # compute pairwise distances between input features and corresponding centers
        centers_batch_bz = torch.stack([centers_batch] * batch_size)
        inputs_bz = torch.stack([inputs] * batch_size).transpose(0, 1)
        if self.l2norm:
            centers_batch_bz = F.normalize(centers_batch_bz, p=2, dim=1)
            inputs_bz = F.normalize(inputs_bz, p=2, dim=1)

        dist = torch.sum((centers_batch_bz - inputs_bz) ** 2, 2).squeeze()
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # for each anchor, find the hardest positive and negative
        mask = targets.expand(batch_size, batch_size).eq(targets.expand(batch_size, batch_size).t())
        dist_ap, dist_an = [], []
        for i in range(batch_size):  # for each sample, we compute distance
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  # mask[i]: positive samples of sample i
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))  # mask[i]==0: negative samples of sample i

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # generate a new label y
        # compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        # y_i = 1, means dist_an > dist_ap + margin will casuse loss be zero
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # prec = (dist_an.data > dist_ap.data).sum().to(dtype=torch.float)/ y.size(0) # normalize data by batch size
        prec = (dist_an.data - dist_ap.data).sum().to(dtype=torch.float) / y.size(0)
        triplet_num = y.shape[0]
        return loss, prec#, triplet_num

if __name__ == "__main__":
    from dataset.TripletSampler import HardestNegativeTripletSelector, RandomNegativeTripletSelector
    margin = 1.
    loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
    batch_size = 12
    embeddings = torch.randn(batch_size, 10)
    target = torch.Tensor([1,1,2,2,3,3,1,1,2,2,3,3]).view(batch_size)
    loss, triplet_num = loss_fn(embeddings, target)
    print(loss, triplet_num)