"""Frozen official DINOv3 B/16 features and locked whole-image biomass heads.

Local contributions are learned gram-equivalent proxies, not measured cell masses.
The D2 regularizer changes the objective only; D1/D2 inference is identical.
"""
from pathlib import Path
import hashlib
import importlib
import re
import sys

import torch
from torch import nn
from torch.nn import functional as F


OFFICIAL_REPOSITORY = 'https://github.com/facebookresearch/dinov3'
DINO_MODEL = 'dinov3_vitb16'
CONSISTENCY_WEIGHT = 0.10
TARGET_WEIGHTS = (0.1, 0.1, 0.1, 0.2, 0.5)


def _local_backbone(repository):
    """Import only the verified backbone, avoiding hubconf's optional segmentors.

    Refuse an already imported foreign ``dinov3`` family rather than silently
    combining packages. Restore sys.path on every outcome and remove only new
    imports on failure; existing modules belonging to other callers are preserved.
    """
    def belongs(module):
        filename = getattr(module, '__file__', None)
        return filename is not None and Path(filename).resolve().is_relative_to(repository)

    before = {name for name in sys.modules if name == 'dinov3' or name.startswith('dinov3.')}
    if any(not belongs(sys.modules[name]) for name in before):
        raise ValueError('A foreign DINOv3 package is already imported; refuse source collision.')
    original_path = list(sys.path)
    sys.path.insert(0, str(repository))
    try:
        backbone = importlib.import_module('dinov3.hub.backbones')
        # pretrained=False constructs locally and cannot fetch checkpoint bytes.
        model = backbone.dinov3_vitb16(pretrained=False)
        loaded = {name for name in sys.modules if name == 'dinov3' or name.startswith('dinov3.')}
        if any(not belongs(sys.modules[name]) for name in loaded):
            raise ValueError('Imported DINOv3 modules escaped the hash-locked repository.')
        return model
    except BaseException:
        for name in list(sys.modules):
            if (name == 'dinov3' or name.startswith('dinov3.')) and name not in before:
                del sys.modules[name]
        raise
    finally:
        sys.path[:] = original_path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_dinov3(repository, checkpoint, metadata, device='cpu'):
    """Load only a hash-locked local official repository and official checkpoint.

    ``metadata`` must contain repository, revision, license, checkpoint_sha256,
    and source_files (relative file -> SHA256). The lock is supplied from the
    independently prepared offline input, never inferred from an arbitrary model.
    All Python files and licenses must be covered by the source lock.
    """
    repository, checkpoint = Path(repository).resolve(), Path(checkpoint).resolve()
    if (metadata.get('repository') != OFFICIAL_REPOSITORY
            or not re.fullmatch(r'[0-9a-f]{40}', metadata.get('revision', ''))
            or not metadata.get('license')
            or metadata.get('model', DINO_MODEL) != DINO_MODEL):
        raise ValueError('A pinned official DINOv3 ViT-B/16 source is required.')
    source_files = metadata.get('source_files', {})
    actual = {str(path.relative_to(repository)): _sha256(path)
              for path in sorted(repository.rglob('*'))
              if path.is_file() and (path.suffix == '.py'
                                    or path.name.upper().startswith('LICENSE'))}
    if not actual or actual != source_files or 'hubconf.py' not in actual:
        raise ValueError('Offline DINOv3 source bytes differ from source lock.')
    checkpoint_hash = _sha256(checkpoint)
    if checkpoint_hash != metadata.get('checkpoint_sha256'):
        raise ValueError('Offline DINOv3 checkpoint SHA256 differs from source lock.')
    model = _local_backbone(repository)
    state = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)
    if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor)
                                            for value in state.values()):
        raise ValueError('Expected the official plain backbone state dictionary.')
    model.load_state_dict(state, strict=True)
    patch_size = getattr(model.patch_embed, 'patch_size', None)
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    storage = getattr(model, 'n_storage_tokens', getattr(model, 'num_register_tokens', None))
    if (getattr(model, 'embed_dim', None) != 768 or tuple(patch_size or ()) != (16, 16)
            or len(model.blocks) != 12 or storage != 4):
        raise ValueError('Loaded backbone is not official DINOv3 ViT-B/16 with four registers.')
    model.requires_grad_(False).eval().to(device)
    provenance = {**metadata, 'checkpoint_sha256': checkpoint_hash,
                  'dim': 768, 'patch_size': 16, 'register_tokens': 4,
                  'resolution': 512, 'spatial_patch_tokens': 1024,
                  'frozen': True, 'strict_state_load': True}
    return model, provenance


def patch_tokens(features, batch):
    """Accept the official patch-only key; concatenated CLS/registers are forbidden."""
    if not isinstance(features, dict) or 'x_norm_patchtokens' not in features:
        raise ValueError('DINOv3 must return its explicit normalized spatial patch tokens.')
    patches = features['x_norm_patchtokens']
    if patches.shape != (batch, 1024, 768):
        raise ValueError('DINOv3 512px patch features must have shape [B,1024,768].')
    if not torch.isfinite(patches).all():
        raise ValueError('Non-finite DINOv3 spatial features.')
    return patches


def dense_grid(patches, images):
    """Row-major 32x32 tokens -> disjoint equal-area 8x8 cells in two views."""
    if patches.shape != (images * 2, 1024, 768):
        raise ValueError('Expected exactly two complete 512px views per original image.')
    grid = patches.reshape(images, 2, 32, 32, 768)
    return grid.reshape(images, 2, 8, 4, 8, 4, 768).mean(dim=(3, 5))


def coarse_grid(fine):
    """Pool neighboring EMBEDDINGS before the shared nonlinear density head."""
    _validate_dense(fine)
    return fine.reshape(len(fine), 2, 4, 2, 4, 2, 768).mean(dim=(3, 5))


@torch.no_grad()
def extract_dense(model, images):
    if images.ndim != 5 or tuple(images.shape[1:]) != (2, 3, 512, 512):
        raise ValueError('Expected [B,2,3,512,512] complete normalized RGB views.')
    if any(parameter.requires_grad for parameter in model.parameters()) or model.training:
        raise ValueError('DINOv3 extraction requires a frozen eval backbone.')
    patches = patch_tokens(model.forward_features(images.flatten(0, 1)), len(images) * 2)
    return dense_grid(patches, len(images))


def _validate_dense(features):
    if features.ndim != 5 or tuple(features.shape[1:]) != (2, 8, 8, 768):
        raise ValueError('Expected dense image cache [B,2,8,8,768].')


def _scale_buffer(scale):
    value = torch.as_tensor(scale, dtype=torch.float32)
    if value.numel() != 1 or not torch.isfinite(value).all() or value.item() <= 0:
        raise ValueError('A finite positive training-fold-only scalar scale is required.')
    return value.reshape(())


def five_targets(primitives):
    if primitives.shape[-1] != 3:
        raise ValueError('Expected Green, Dead, Clover primitives.')
    gdm = primitives[..., 0] + primitives[..., 2]
    total = gdm + primitives[..., 1]
    return torch.cat((primitives, gdm.unsqueeze(-1), total.unsqueeze(-1)), dim=-1)


class DINOGlobalHead(nn.Module):
    """D0: global pooling followed by a small nonlinear image-level head."""
    def __init__(self, scale):
        super().__init__()
        self.register_buffer('scale', _scale_buffer(scale))
        self.head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256),
                                  nn.GELU(), nn.Linear(256, 3))

    def forward(self, features):
        _validate_dense(features)
        pooled = features.float().mean(dim=(1, 2, 3))
        return five_targets(F.softplus(self.head(pooled).float()) * self.scale)


class DINODensityHead(nn.Module):
    """D1 and D2 use this SAME shared nonlinear 768->256->3 head."""
    def __init__(self, scale):
        super().__init__()
        self.register_buffer('scale', _scale_buffer(scale))
        self.density_head = nn.Sequential(nn.Linear(768, 256), nn.GELU(), nn.Linear(256, 3))

    def local_contributions(self, features):
        _validate_dense(features)
        return F.softplus(self.density_head(features.float()).float()) * self.scale

    def forward(self, features):
        return five_targets(self.local_contributions(features).mean(dim=(1, 2, 3)))

    def predictions_at_scales(self, features):
        fine = self(features)
        coarse = F.softplus(self.density_head(coarse_grid(features).float()).float()) * self.scale
        return fine, five_targets(coarse.mean(dim=(1, 2, 3)))


def make_head(arm, scale):
    if arm == 'D0':
        return DINOGlobalHead(scale)
    if arm in ('D1', 'D2'):
        return DINODensityHead(scale)
    raise ValueError('DINO arm must be D0, D1 or D2.')


def consistency_penalty(pred_fine, pred_coarse, scale):
    if pred_fine.shape != pred_coarse.shape or pred_fine.ndim != 2 or pred_fine.shape[1] != 5:
        raise ValueError('Scale consistency compares aggregated whole-image predictions [B,5].')
    return ((pred_fine[:, :3].float() - pred_coarse[:, :3].float()) ** 2).mean() / scale ** 2


def head_loss(head, features, labels, arm):
    """One supervised label per original image; D2 adds only the locked regularizer."""
    if arm not in ('D0', 'D1', 'D2'):
        raise ValueError('DINO arm must be D0, D1 or D2.')
    if labels.shape != (len(features), 5):
        raise ValueError('Labels must be whole-image [B,5], never per-cell targets.')
    if arm == 'D2':
        if not isinstance(head, DINODensityHead):
            raise ValueError('D2 must share the D1 density-head architecture.')
        pred, coarse = head.predictions_at_scales(features)
        penalty = consistency_penalty(pred, coarse, head.scale)
    else:
        pred, coarse = head(features), None
        penalty = pred.new_zeros(())
    weights = pred.new_tensor(TARGET_WEIGHTS)
    supervised = ((pred.float() - labels.float()) ** 2 * weights).sum(dim=-1).mean() / head.scale ** 2
    loss = supervised + CONSISTENCY_WEIGHT * penalty
    return loss, {'pred_fine': pred, 'pred_coarse': coarse,
                  'supervised_loss': supervised, 'consistency_loss': penalty}


def nondegeneracy_proof(head, features):
    """Mandatory executable gate: consistency alone must affect density parameters."""
    fine, coarse = head.predictions_at_scales(features)
    penalty = consistency_penalty(fine, coarse, head.scale)
    grads = torch.autograd.grad(penalty, tuple(head.density_head.parameters()), allow_unused=False)
    maximum = max(float(grad.detach().abs().max()) for grad in grads)
    if not torch.isfinite(penalty) or penalty.item() <= 1e-8 or not maximum > 0:
        raise ValueError('DINO cross-scale objective failed its nondegeneracy gate.')
    if not all(torch.isfinite(grad).all() for grad in grads):
        raise ValueError('Non-finite cross-scale density gradient.')
    return {'penalty': float(penalty.detach()), 'gradient_max_abs': maximum,
            'fine_coarse_max_difference': float((fine - coarse).detach().abs().max()),
            'trainable_parameters': sum(p.numel() for p in head.parameters() if p.requires_grad)}
