import torch
from torch import nn
from torch.nn import functional
import numpy as np


# From https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/losses.py
class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = np.cos(margin)
        self.sin_m = np.sin(margin)
        self.theta = np.cos(np.pi - margin)
        self.sinmm = np.sin(np.pi - margin) * margin
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s   
        
        return logits