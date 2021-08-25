import torch
from torch import nn

class STDLayerNorm(nn.Module):
    """Construct a layernorm module"""
    def __init__(self, num_features: int, eps=1e-6):
        super(STDLayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(num_features), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

"""
https://github.com/bzhangGo/rmsnorm
@inproceedings{zhang-sennrich-neurips19,
    address = "Vancouver, Canada",
    author = "Zhang, Biao and Sennrich, Rico",
    booktitle = "Advances in Neural Information Processing Systems 32",
    url = "https://openreview.net/references/pdf?id=S1qBAf6rr",
    title = "{Root Mean Square Layer Normalization}",
    year = "2019"
}
"""
class RMSLayerNorm(nn.Module):
    def __init__(self, num_features: int, p=-1., eps=1e-8, bias=False):
        super(RMSLayerNorm, self).__init__()
        self.eps = eps
        self.d = num_features
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(num_features))
        self.register_parameter("scale", self.scale)
        if self.bias:
            self.offset = nn.Parameter(torch.zeros(num_features))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed