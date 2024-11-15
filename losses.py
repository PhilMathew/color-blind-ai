import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class CrossEntropyLoss(nn.Module):
    # We reimplement this to play nicely with the training paradigm for facial recognition
    # and to allow us to switch between it and ArcFace easily. The only difference is that
    # this loss applies the linear layer instead of doing it outside the function.
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
    
    def forward(self, reps: torch.Tensor, labels: torch.Tensor, fc_weight: torch.Tensor):
        logits = F.linear(reps, fc_weight)
        
        return F.cross_entropy(logits, labels)


class ArcFaceLoss(nn.Module):
    """ 
    ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s: float = 64.0, margin: float = 0.5, easy_margin: bool = False):
        super(ArcFaceLoss, self).__init__()
        self.scale = s
        self.cos_m = np.cos(margin)
        self.sin_m = np.sin(margin)
        self.theta = np.cos(np.pi - margin)
        self.sinmm = np.sin(np.pi - margin) * margin
        self.easy_margin = easy_margin

    def forward(self, reps: torch.Tensor, labels: torch.Tensor, fc_weight: torch.Tensor):
        # Normalize features and weights
        reps_norm = F.normalize(reps)
        fc_weight_norm = F.normalize(fc_weight)
        
        # Apply normalized weights to get the logits for ArcFace loss
        logits = F.linear(reps_norm, fc_weight_norm)
        logits = logits.clamp(-1, 1)
        
        # ArcFace loss
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index]]

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)

        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.scale
        
        # Apply cross entropy on ArcFace output
        loss = F.cross_entropy(logits, labels)
        
        return loss
