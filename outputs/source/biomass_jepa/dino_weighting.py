"""One controlled follow-up: learned pooling, not measured local mass or area.

Uses exactly the saved frozen DINO B/16 dense cache from the completed study.
The D1 density trunk and its initialization are unchanged. Per-target softmax
weights sum to one over all 128 cells in a complete two-view photograph.
"""
import torch
from torch import nn
from .dinov3_dense import DINODensityHead, five_targets, TARGET_WEIGHTS


class DINOWeightedHead(DINODensityHead):
    def __init__(self, scale):
        super().__init__(scale)
        # Do not consume the RNG used by the shared density trunk/data order.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(314159)
            self.weight_head = nn.Sequential(nn.Linear(768, 128), nn.GELU(), nn.Linear(128, 3))
            nn.init.zeros_(self.weight_head[-1].weight)
            nn.init.zeros_(self.weight_head[-1].bias)

    def pooling_weights(self, features):
        logits = self.weight_head(features.float()).float().reshape(len(features), 128, 3)
        return torch.softmax(logits, dim=1).reshape(len(features), 2, 8, 8, 3)

    def forward(self, features):
        local = self.local_contributions(features)
        return five_targets((local * self.pooling_weights(features)).sum(dim=(1, 2, 3)))


def make_head(arm, scale):
    if arm == 'D1':
        return DINODensityHead(scale)
    if arm == 'D3':
        return DINOWeightedHead(scale)
    raise ValueError('This controlled study supports only D1 and D3.')


def head_loss(head, features, labels, arm):
    if arm not in ('D1', 'D3') or labels.shape != (len(features), 5):
        raise ValueError('Whole-photograph labels and D1/D3 required.')
    pred = head(features)
    loss = ((pred.float()-labels.float()).square()*pred.new_tensor(TARGET_WEIGHTS)).sum(1).mean()/head.scale.square()
    return loss, {'pred_fine':pred, 'consistency_loss':pred.new_zeros(())}
