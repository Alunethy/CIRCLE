"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 12/12/2021
"""
from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
import gc
import math

class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class Circulate_NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(Circulate_NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size
        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j, label):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0).cpu()
        label = torch.cat((label, label), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = (self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature).cuda()

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.cuda()).long()
        # logits = torch.cat((positive_samples, negative_samples), dim=1)

        dirc = {}
        for la in label:
            dirc[str(la)] = la

        loss_pos = torch.tensor(0.0).cuda()

        for key in dirc:
            # get the neg for each class
            dirction_eq = np.where(label.cpu() == dirc[key].cpu())
            list_dirction_eq = list(dirction_eq[0])
            dirction_not_eq = np.where(label.cpu() != dirc[key].cpu())
            list_dirction_not_eq = list(dirction_not_eq[0])

            mat = sim[list_dirction_eq]
            neg = mat[:, list_dirction_not_eq]

            labels_dirc = labels[list_dirction_eq].squeeze(0)

            pos = positive_samples[list_dirction_eq, :].squeeze(0)

            logits = torch.cat((pos, neg), dim=1)

            loss = self.criterion(logits, labels_dirc)
            loss /= len(labels_dirc)
            loss_pos += loss
        return {"loss": loss_pos}

class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size
        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j, label):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0).cpu()
        # label = torch.cat((label, label), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        loss = self.criterion(logits, labels)
        loss /= N
        return {"loss": loss}


class PairConLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(PairConLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-08
        print(f"\n Initializing PairConLoss \n")

    def forward(self, features_1, features_2, temp=None):
        device = features_1.device
        batch_size = features_1.shape[0]
        features = torch.cat([features_1, features_2], dim=0)
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = mask.repeat(2, 2)
        mask = ~mask
        
        pos = torch.exp(torch.sum(features_1*features_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)
        neg = torch.exp(torch.mm(features, features.t().contiguous()) / self.temperature)
        neg = neg.masked_select(mask).view(2*batch_size, -1)
        
        neg_mean = torch.mean(neg)
        pos_n = torch.mean(pos)
        Ng = neg.sum(dim=-1)
            
        loss_pos = (- torch.log(pos / (Ng+pos))).mean()
        
        return {"loss":loss_pos, "pos_mean":pos_n.detach().cpu().numpy(), "neg_mean":neg_mean.detach().cpu().numpy(), "pos":pos.detach().cpu().numpy(), "neg":neg.detach().cpu().numpy()}

# positive pair will pull, but neg pairs(other cluster) will push
class CirculatePairConLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(CirculatePairConLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-08
        print(f"\n Initializing CirculatePairConLoss \n")

    def forward(self, f1, f2, label):
        device = label.device
        loss_pos = torch.tensor(0.0).to(device)

        dirc = {}
        for la in label:
            dirc[str(la)] = la

        for key in dirc:
            # get the neg for each class
            dirction_eq = np.where(label.cpu() == dirc[key].cpu())
            dirction_not_eq = np.where(label.cpu() != dirc[key].cpu())
            pos_f1, pos_f2 = f1[dirction_eq, :].squeeze(0), f2[dirction_eq, :].squeeze(0)
            neg_f1, neg_f2 = f1[dirction_not_eq, :].squeeze(0), f2[dirction_not_eq, :].squeeze(0)

            pos_features = torch.cat([pos_f1, pos_f2], dim=0)
            neg_features = torch.cat([neg_f1, neg_f2], dim=0)
            neg = torch.exp(torch.mm(pos_features, neg_features.t().contiguous()) / self.temperature)
            Ng = neg.sum(dim=-1)

            pos = torch.exp(torch.sum(pos_f1 * pos_f2, dim=-1) / self.temperature)
            pos = torch.cat([pos, pos], dim=0)

            loss_pos += (- torch.log(pos / (Ng + pos))).mean()
        return {"loss": loss_pos}


class HypersphereLoss(nn.Module):
    def __init__(self):
        super(HypersphereLoss, self).__init__()
        self.alpha = 2
        self.t = 2
        self.eps = 1e-08
        print(f"\n Initializing CirculatePairConLoss \n")

    def forward(self, x, y):
        align_loss = (x - y).norm(p=2, dim=1).pow(self.alpha).mean()

        uniform_loss = torch.pdist(x, p=2).pow(2).mul(-self.t).exp().mean().log()

        uniform_loss_y = torch.pdist(y, p=2).pow(2).mul(-self.t).exp().mean().log()

        loss_pos = align_loss + uniform_loss + uniform_loss_y
        return {"loss": loss_pos}


class SingleCirculatePairConLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(SingleCirculatePairConLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-08
        print(f"\n Initializing SingleCirculatePairConLoss \n")

    def forward(self, f1, label):

        loss_pos = torch.tensor(0.0).cuda()

        dirc = {}
        for la in label:
            dirc[str(la)] = la

        for key in dirc:
            # get the neg for each class
            dirction_eq = np.where(label.cpu() == dirc[key].cpu())
            dirction_not_eq = np.where(label.cpu() != dirc[key].cpu())
            pos_f1 = f1[dirction_eq, :].squeeze(0)
            neg_f1 = f1[dirction_not_eq, :].squeeze(0)

            pos_features = pos_f1
            neg_features = neg_f1
            neg = torch.exp(torch.mm(pos_features, neg_features.t().contiguous()) / self.temperature)
            # mask = torch.eye(batch_size, dtype=torch.bool).to(device)
            # mask = mask.repeat(2, 2)
            # mask = ~mask
            # neg = torch.exp(torch.mm(features, features.t().contiguous()) / self.temperature).masked_select(mask).view(
            #     2 * batch_size, -1)

            # neg_mean = torch.mean(neg)
            # pos_n = torch.mean(pos)

            Ng = neg.sum(dim=-1)

            # features_1, features_2 = f1[dirction, :].squeeze(0), f2[dirction, :].squeeze(0)
            pos_con = torch.exp(torch.mm(pos_features, pos_features.t().contiguous()) / self.temperature)
            pos = pos_con.sum(dim=-1)
            # pos = torch.cat([pos, pos], dim=0)

            loss_pos += (- torch.log(pos / (Ng + pos))).mean()

        return {"loss": loss_pos}


class Neighbor_PairConLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(Neighbor_PairConLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-08
        print(f"\n Initializing CirculatePairConLoss \n")

    def forward(self, feat1, feat2, indices1, indices2):
        nerest_data1 = feat1[indices1]
        nerest_data2 = feat2[indices1]
        largest_data1 = feat1[indices2]
        largest_data2 = feat2[indices2]

        device = feat1.device
        loss_pos = torch.tensor(0.0).to(device)

        # get the neg for each class
        pos_f1, pos_f2 = nerest_data1, nerest_data2
        neg_f1, neg_f2 = largest_data1, largest_data2

        pos_features = torch.cat([pos_f1, pos_f2], dim=0)
        neg_features = torch.cat([neg_f1, neg_f2], dim=0)
        neg = torch.exp(torch.mm(pos_features, neg_features.t().contiguous()) / self.temperature)
        Ng = neg.sum(dim=-1)

        pos = torch.exp(torch.mm(pos_features, pos_features.t().contiguous()) / self.temperature)
        Pg = pos.sum(dim=-1)

        loss_pos += (- torch.log(Pg / (Ng + Pg))).mean()
        return {"loss": loss_pos}


class central_PairConLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(central_PairConLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-08

    def forward(self, f1, label, centrals):
        device = label.device
        loss_pos = torch.tensor(0.0).to(device)

        dirc = {}
        for la in label:
            dirc[str(la)] = la

        for key in dirc:
            central = centrals[dirc[key].cpu()]  # get the central of this class
            # get the neg for each class
            dirction_eq = np.where(label.cpu() == dirc[key].cpu())
            dirction_not_eq = np.where(label.cpu() != dirc[key].cpu())
            # 获得当前类别的所有数据点
            pos_f1 = f1[dirction_eq, :].squeeze(0)
            # neg_f1 = f1[dirction_not_eq, :].squeeze(0)

            central_pos = central.repeat(pos_f1.shape[0], 1)  # repeat the central

            pos = torch.exp(torch.sum(pos_f1 * central_pos, dim=-1) / self.temperature)

            Ng_all = pos
            for key2 in dirc:
                if key == key2:
                    continue
                else:
                    # 获得其他类别的质心 与 上一层类别数据的距离
                    central_neg = centrals[dirc[key2].cpu()]
                    central_neg = central_neg.repeat(pos_f1.shape[0], 1)

                    neg = torch.exp(torch.mm(pos_f1, central_neg.t().contiguous()) / self.temperature)
                    Ng = neg.sum(dim=-1)
                    # Ng_all = Ng_all + Ng
                    loss_pos += (- torch.log(pos / (Ng_all + Ng))).mean()



        return {"loss": loss_pos}


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss


class SupConLoss(nn.Module):
    """It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return {"loss": loss}


if __name__ == '__main__':
    # contrast_loss = SupConLoss()
    # x1 = torch.rand(100, 1, 64).cuda()
    # central = torch.rand(3, 64).cuda()
    # y = np.random.randint(0, 3, size=100)
    # f = contrast_loss(x1, torch.tensor(y).cuda())
    # print(f)
    temp = torch.rand(10, 3).cuda()
    max_value, max_index = torch.max(temp, dim=1)
    print(max_index)
    print(max_value)

    values, indices = max_value.topk(5, dim=0, largest=True, sorted=False)
    print(indices)
    print(values)

    print(max_index[indices])






