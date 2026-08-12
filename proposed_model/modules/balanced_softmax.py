import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedSoftmaxLoss(nn.Module):

    def __init__(self, samples_per_class):
        super().__init__()

        samples = torch.tensor(
            samples_per_class,
            dtype=torch.float32,
        )

        self.register_buffer("log_prior", torch.log(samples / samples.sum()))

    def forward(self, logits, targets):

        logits = logits + self.log_prior

        return F.cross_entropy(
            logits,
            targets,
        )
