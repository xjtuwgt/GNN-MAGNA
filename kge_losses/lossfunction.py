from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import _reduction as _Reduction

class KGESmoothCELoss(nn.Module):
    def __init__(self, smoothing=0.001, mode='multiply'):
        super(KGESmoothCELoss, self).__init__()
        self.loss_function = CESmoothLossOnevsAll(smoothing=smoothing)
        self.mode = mode

    def forward(self, head_emb, tail_emb, all_rel_emb, labels):
        if self.mode == 'multiply':
            ent_emb = head_emb * tail_emb
        elif self.mode == 'trans':
            ent_emb = head_emb - tail_emb
        else:
            raise ValueError('Training batch mode %s not supported' % self.mode)
        scores = torch.matmul(ent_emb, all_rel_emb.transpose(1,0))
        loss = self.loss_function(scores, labels)
        return loss


class CESmoothLossKvsAll(nn.Module):
    def __init__(self, smoothing=0.0):
        super(CESmoothLossKvsAll, self).__init__()
        self.smoothing = smoothing
        self.kl_criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, predictions, bi_labels=None):
        log_prob = torch.log_softmax(predictions, dim=-1)
        if bi_labels is None:
            return torch.tensor([0.0])
        with torch.no_grad():
            weights = bi_labels * (1 - self.smoothing) + (self.smoothing / bi_labels.size(-1))
        weights = F.normalize(weights, p=1, dim=1)
        loss = self.kl_criterion(log_prob, weights)
        return loss

class CESmoothLossOnevsAll(nn.Module):
    def __init__(self, smoothing=0.0):
        super(CESmoothLossOnevsAll, self).__init__()
        self.smoothing = smoothing
        self.kl_criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, predictions, labels):
        log_prob = torch.log_softmax(predictions, dim=-1)
        with torch.no_grad():
            weight = predictions.new_ones(predictions.size()) * self.smoothing / (predictions.size(-1) - 1.)
            weight.scatter_(-1, labels.unsqueeze(-1), (1. - self.smoothing))
        loss = self.kl_criterion(log_prob, weight)
        return loss

class BCESmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(BCESmoothLoss, self).__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, predictions, bi_labels=None):
        if bi_labels is None:
            return torch.tensor([0.0])
        smooth_labels = bi_labels * (1-self.smoothing) + (self.smoothing/bi_labels.size(-1))
        loss = self.bce(predictions, smooth_labels)
        return loss


class BCESmoothLossOnevsAll(nn.Module):
    def __init__(self, smoothing=0.0):
        super(BCESmoothLossOnevsAll, self).__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, predictions, labels):
        with torch.no_grad():
            weight = predictions.new_ones(predictions.size()) * self.smoothing / (predictions.size(-1) - 1.)
            weight.scatter_(-1, labels.unsqueeze(-1), (1. - self.smoothing))
        loss = self.bce(predictions, weight)
        return loss

####++++++++++++++++++++++++++++++++++++
def weight_kl_div(input, target, weights = None, reduction='batchmean'):
    reduction_enum = _Reduction.get_enum('sum')
    reduced = torch.kl_div(input, target, reduction_enum)
    if reduction == 'batchmean' and input.dim() != 0 and weights is None:
        reduced = reduced / input.size()[0]
    elif reduction == 'batchmean' and weights is not None:
        reduced = reduced * weights
    return reduced

class WeightedKLDivLoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(WeightedKLDivLoss, self).__init__()
        self.reduction=reduction

    def forward(self, input, target, weights=None):
        return weight_kl_div(input, target, reduction=self.reduction)

class WeightedCESmoothLossKvsAll(nn.Module):
    def __init__(self, smoothing=0.0):
        super(WeightedCESmoothLossKvsAll, self).__init__()
        self.smoothing = smoothing
        self.kl_criterion = WeightedKLDivLoss(reduction='batchmean')

    def forward(self, predictions, bi_labels=None, samp_weights=None):
        log_prob = torch.log_softmax(predictions, dim=-1)
        if bi_labels is None:
            return torch.tensor([0.0])
        with torch.no_grad():
            weights = bi_labels * (1 - self.smoothing) + (self.smoothing / bi_labels.size(-1))
        weights = F.normalize(weights, p=1, dim=1)
        loss = self.kl_criterion(log_prob, weights, samp_weights)
        return loss

class WeightedCESmoothLossOnevsAll(nn.Module):
    def __init__(self, smoothing=0.0):
        super(WeightedCESmoothLossOnevsAll, self).__init__()
        self.smoothing = smoothing
        self.kl_criterion = WeightedKLDivLoss(reduction='batchmean')

    def forward(self, predictions, labels, samp_weights=None):
        log_prob = torch.log_softmax(predictions, dim=-1)
        with torch.no_grad():
            weight = predictions.new_ones(predictions.size()) * self.smoothing / (predictions.size(-1) - 1.)
            weight.scatter_(-1, labels.unsqueeze(-1), (1. - self.smoothing))
        loss = self.kl_criterion(log_prob, weight, samp_weights)
        return loss

# if __name__ == '__main__':
#     x1 = torch.ones(3, 4)
#     x2 = torch.ones(3, 4)
#     x3 = torch.zeros(5, 4)
#     y = torch.ones(3).long()
#
#     cos_emb_loss = KGESmoothCELoss()
#     loss = cos_emb_loss(x1, x2, x3, y)
#     print(loss)