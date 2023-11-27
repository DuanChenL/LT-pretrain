import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class CosineClassifier(nn.Module):
    def __init__(self):
        super(CosineClassifier, self).__init__()
        self.cosine_layer = nn.Linear(in_features=1024, out_features=1231, bias=False)

    def forward(self, x):
        # s = (x.norm(dim=-1, keepdim=True) * self.cosine_layer.weight.norm(dim=-1, keepdim=True).T)
        # s = s.sum(dim=-1)
        # s = s.mean()
        s = 20
        Wstar = self.cosine_layer.weight / self.cosine_layer.weight.norm(dim=-1, keepdim=True)
        # norm = x.norm(dim=-1, keepdim=True)
        # norm_2 = x.norm(dim=0, keepdim=True)
        x = x / x.norm(dim=-1, keepdim=True)

        # res_1 = torch.matmul(x, Wstar.T)
        res = s * torch.matmul(x, Wstar.T)
        # res_2 = 0
        return res