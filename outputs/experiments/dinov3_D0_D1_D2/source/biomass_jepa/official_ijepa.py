"""Adapters for the matching official I-JEPA encoder and latent predictor."""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from .ijepa_reference import vision_transformer as reference
from .ijepa_reference.masks import apply_masks
from .ijepa_reference.tensors import repeat_interleave_batch
from .models import BiomassModel, complementary_patch_masks

OFFICIAL_MODEL = "official_ijepa_vith14"
OFFICIAL_REVISION = "52c1ae95d05f743e000e8f10a1f3a79b10cff048"


def strip_ddp(state):
    """Official full checkpoints contain DistributedDataParallel key prefixes."""
    if not state or not all(key.startswith("module.") for key in state):
        raise ValueError("Expected the original official DDP state namespace.")
    return {key.removeprefix("module."): value for key, value in state.items()}


class OfficialEncoder(nn.Module):
    def __init__(self, size=224, last_blocks=8, tiny=False):
        super().__init__()
        if size != 224 and not tiny:
            raise ValueError("This official checkpoint experiment retains its 224px grid.")
        if tiny:
            self.backbone = reference.VisionTransformer(img_size=[size], patch_size=14,
                                                       embed_dim=64, depth=2, num_heads=4)
        else:
            self.backbone = reference.vit_huge(img_size=[size], patch_size=14)
        self.dim = self.backbone.embed_dim
        self.grid = size//14
        self.tiny = tiny
        self.checkpoint_blocks = False
        self.configure_trainable(last_blocks)

    def configure_trainable(self, last_blocks):
        if not 0 <= last_blocks <= len(self.backbone.blocks):
            raise ValueError("Official trainable scope exceeds the actual encoder depth.")
        self.requires_grad_(False)
        if last_blocks:
            for block in self.backbone.blocks[-last_blocks:]:
                block.requires_grad_(True)
            self.backbone.norm.requires_grad_(True)

    def forward(self, images, masks=None):
        if images.shape[-2:] != (self.grid*14, self.grid*14):
            raise ValueError("Official input size must match the checkpoint patch grid.")
        if masks is not None and isinstance(masks, torch.Tensor):
            masks = [masks]
        return self.backbone(images, masks)


def make_predictor(encoder):
    return reference.vit_predictor(num_patches=encoder.grid**2, embed_dim=encoder.dim,
                                   predictor_embed_dim=32 if encoder.tiny else 384,
                                   depth=2 if encoder.tiny else 12,
                                   num_heads=encoder.backbone.num_heads)


def load_official_models(checkpoint, *, last_blocks=8, size=224, tiny=False):
    if not all(key in checkpoint for key in ("encoder", "predictor", "target_encoder")):
        raise ValueError("Need the matching official student, predictor and EMA teacher.")
    encoder = OfficialEncoder(size, last_blocks, tiny)
    predictor = make_predictor(encoder)
    teacher = copy.deepcopy(encoder)
    encoder.backbone.load_state_dict(strip_ddp(checkpoint["encoder"]), strict=True)
    predictor.load_state_dict(strip_ddp(checkpoint["predictor"]), strict=True)
    teacher.backbone.load_state_dict(strip_ddp(checkpoint["target_encoder"]), strict=True)
    teacher.requires_grad_(False)
    teacher.eval()
    return encoder, predictor, teacher


def ssl_latents(encoder, predictor, teacher, images, context_masks, target_masks, *, masks_validated=False):
    """The official context/target ordering and normalized teacher objective."""
    if not masks_validated:
        validate_ssl_masks(context_masks, target_masks, len(images), encoder.grid**2)
    with torch.no_grad():
        teacher.eval()
        features = teacher(images)
        targets = F.layer_norm(features, (features.shape[-1],))
        targets = apply_masks(targets, target_masks)
        targets = repeat_interleave_batch(targets, len(images), repeat=len(context_masks))
    contexts = encoder(images, context_masks)
    predictions = predictor(contexts, context_masks, target_masks)
    if predictions.shape != targets.shape:
        raise ValueError("Official latent target ordering or shape mismatch.")
    return predictions, targets


def validate_ssl_masks(context_masks, target_masks, batch_size, patches):
    if not context_masks or not target_masks:
        raise ValueError("Need nonempty context and target mask lists.")
    for masks in (context_masks, target_masks):
        if len({mask.shape[1] for mask in masks if mask.ndim == 2}) != 1:
            raise ValueError("Official masks in each list must have equal token counts.")
        for mask in masks:
            if (mask.ndim != 2 or len(mask) != batch_size or mask.dtype != torch.long
                    or mask.shape[1] == 0 or mask.min() < 0 or mask.max() >= patches):
                raise ValueError("Invalid official per-image mask indices.")
            if any(len(torch.unique(row)) != len(row) for row in mask):
                raise ValueError("A mask repeats a patch index.")
    for context in context_masks:
        for target in target_masks:
            if (context[:, :, None] == target[:, None, :]).any():
                raise ValueError("A target patch appears in its per-image context.")


class OfficialBiomassModel(BiomassModel):
    """A new labeled head reads actual official predictor query outputs."""
    def __init__(self, encoder, predictor, *, hidden=384, auxiliary_bins=7,
                 dropout=.2, output_space="log1p"):
        nn.Module.__init__(self)
        self.encoder, self.predictor = encoder, predictor
        self.output_space = output_space
        self.attention = nn.Linear(encoder.dim, 1)
        self.readout = nn.Sequential(nn.LayerNorm(encoder.dim), nn.Linear(encoder.dim, hidden),
                                     nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 3))
        self.auxiliary_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(encoder.dim), nn.Linear(encoder.dim, hidden//2),
                          nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden//2, auxiliary_bins))
            for _ in range(5)]) if auxiliary_bins else nn.ModuleList()

    def patch_predictions(self, images, detach_encoder=False):
        predictions, positions = [], []
        for visible, target in complementary_patch_masks(self.encoder.grid, images.device):
            context_mask = visible.expand(len(images), -1)
            target_mask = target.expand(len(images), -1)
            if detach_encoder:
                with torch.no_grad():
                    context = self.encoder(images, [context_mask])
            else:
                context = self.encoder(images, [context_mask])
            predictions.append(self.predictor(context, [context_mask], [target_mask]))
            positions.append(target)
        return torch.cat(predictions, dim=1)[:, torch.cat(positions).argsort()]

    def forward(self, tiles, return_auxiliary=False, detach_encoder=False):
        batch, views, channels, height, width = tiles.shape
        images = tiles.reshape(batch*views, channels, height, width)
        features = self.patch_predictions(images, detach_encoder).mean(dim=1)
        features = features.reshape(batch, views, -1)
        weights = self.attention(features).float().softmax(dim=1).to(features.dtype)
        pooled = (weights*features).sum(dim=1)
        primitive = F.softplus(self.readout(pooled).float())
        if self.output_space == "log1p":
            primitive = torch.expm1(primitive)
        green, dead, clover = primitive.unbind(dim=1)
        mass = torch.stack((green, dead, clover, green+clover, green+clover+dead), dim=1)
        if return_auxiliary:
            if not self.auxiliary_heads:
                raise ValueError("No auxiliary training heads were configured.")
            return mass, torch.stack([head(pooled) for head in self.auxiliary_heads], dim=1)
        return mass


def encoder_factory(args, size, pretrained):
    if pretrained:
        raise ValueError("Official pretrained weights must be supplied explicitly from the pinned checkpoint.")
    return OfficialEncoder(size=size, last_blocks=args.last_blocks, tiny=args.tiny)


def regression_factory(encoder, predictor_width, predictor_depth, output_space,
                       pretrained_predictor=None, auxiliary_bins=0, predictor_dropout=0.,
                       head_dropout=.1, regression_hidden=None, segmentation_classes=0,
                       independent_targets=False, predictor_readout="context", observed_latent_weight=0.):
    expected = (32, 2) if encoder.tiny else (384, 12)
    if ((predictor_width, predictor_depth) != expected or pretrained_predictor is None
            or predictor_dropout or segmentation_classes or independent_targets
            or observed_latent_weight or predictor_readout != "masked_latent"):
        raise ValueError("Official predictor requires its full matching query/projection state and unchanged architecture.")
    if regression_hidden is not None and (len(regression_hidden) != 1 or
                                         (not encoder.tiny and regression_hidden != [384])):
        raise ValueError("The official controlled experiment uses one 384-wide regression hidden layer.")
    predictor = make_predictor(encoder)
    predictor.load_state_dict(pretrained_predictor, strict=True)
    hidden = regression_hidden[0] if regression_hidden else predictor_width
    return OfficialBiomassModel(encoder, predictor, hidden=hidden, auxiliary_bins=auxiliary_bins,
                                dropout=head_dropout, output_space=output_space)
