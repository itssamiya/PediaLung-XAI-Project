import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss

    Parameters
    ----------
    gamma : float
        Focusing parameter.
    alpha : Tensor or None
        Optional class weights.
    reduction : str
        'mean' or 'sum'
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):

        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        pt = torch.exp(-ce_loss)

        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            focal_loss = alpha[targets] * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()

        elif self.reduction == "sum":
            return focal_loss.sum()

        return focal_loss
