import copy
import math

import timm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class Encoder(nn.Module):
    def __init__(self, model_name, size, pretrained=True, last_blocks=8, tiny=False):
        super().__init__()
        if tiny:
            from timm.models.eva import Eva
            self.backbone = Eva(img_size=size, patch_size=16, embed_dim=64, depth=2,
                                num_heads=4, num_reg_tokens=4, use_abs_pos_emb=False,
                                use_rot_pos_emb=True, rope_type="dinov3", rope_rotate_half=True,
                                dynamic_img_size=True, num_classes=0)
        else:
            self.backbone = timm.create_model(model_name, pretrained=pretrained,
                                             num_classes=0, img_size=size, dynamic_img_size=True)
        if not hasattr(self.backbone, "rope") or getattr(self.backbone, "rope_mixed", False):
            raise ValueError("Runner requires timm's EVA implementation of axial DINOv3.")
        self.dim = self.backbone.embed_dim
        self.grid = size // 16
        self.checkpoint_blocks = False
        self.configure_trainable(last_blocks)

    def configure_trainable(self, last_blocks):
        self.requires_grad_(False)
        if last_blocks:
            for block in self.backbone.blocks[-last_blocks:]:
                block.requires_grad_(True)
            self.backbone.norm.requires_grad_(True)

    def forward(self, image, visible=None, return_summary=False):
        b = self.backbone
        x = b.patch_embed(image)
        x, rope = b._pos_embed(x)
        prefix = b.num_prefix_tokens
        if visible is not None:
            # Remove target patches BEFORE any attention. Full-image features cannot be masked post hoc.
            x = torch.cat([x[:, :prefix], x[:, prefix:][:, visible]], dim=1)
            if rope is not None:
                if rope.ndim != 2:
                    raise ValueError(f"Unexpected axial RoPE shape: {rope.shape}")
                rope = rope[visible]
        x = b.norm_pre(x)
        for block in b.blocks:
            if self.checkpoint_blocks and self.training and any(p.requires_grad for p in block.parameters()):
                x = checkpoint(block, x, rope=rope, use_reentrant=False)
            else:
                x = block(x, rope=rope)
        features = b.norm(x)
        if return_summary:
            if visible is not None:
                raise ValueError('The plain DINO summary requires all image patches.')
            return torch.cat([features[:, 0], features[:, prefix:].mean(dim=1)], dim=-1)
        return features[:, prefix:]


def block_mask(grid, device, targets=4, fraction=.15):
    side = max(1, min(grid-1, round(grid * math.sqrt(fraction))))
    mask = torch.zeros(grid, grid, dtype=torch.bool, device=device)
    for _ in range(targets):
        top, left = torch.randint(grid-side+1, (2,), device=device).tolist()
        mask[top:top+side, left:left+side] = True
    if mask.all() or not mask.any():
        raise RuntimeError("Mask has no visible or no target tokens.")
    indices = torch.arange(grid*grid, device=device)
    return indices[~mask.flatten()], indices[mask.flatten()]


def complementary_patch_masks(grid, device):
    """Two deterministic spatial contexts, predicting every patch once without seeing it.

    Each pass predicts one vertical half from the other half. Both passes are used
    in training and evaluation; no test-time randomness or target pixels in context.
    """
    if grid < 2:
        raise ValueError("Masked-latent readout needs at least two patch columns.")
    indices = torch.arange(grid * grid, device=device)
    left = indices[indices.remainder(grid) < grid // 2]
    right = indices[indices.remainder(grid) >= grid // 2]
    return ((right, left), (left, right))


class PredictorBlock(nn.Module):
    def __init__(self, width, heads=6):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3*width), nn.Linear(width, width)
        self.fc1, self.fc2 = nn.Linear(width, 4*width), nn.Linear(4*width, width)
        self.dropout = nn.Dropout(0.)

    def forward(self, x):
        batch, tokens, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, tokens, 3, self.heads, width//self.heads).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v,
            dropout_p=self.dropout.p if self.training else 0.).transpose(1, 2).reshape(batch, tokens, width)
        x = x + self.dropout(self.proj(attended))
        return x + self.dropout(self.fc2(F.gelu(self.fc1(self.norm2(x)))))


class Predictor(nn.Module):
    def __init__(self, encoder_dim, grid, width=384, depth=6):
        super().__init__()
        if width < 4 or width % 4:
            raise ValueError("Predictor width must be a positive multiple of four for 2D positions and attention.")
        self.input_proj = nn.Linear(encoder_dim, width)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        yy, xx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
        frequencies = 1 / 10000 ** (torch.arange(width//4).float() / (width//4))
        pos = torch.cat([f.flatten()[:, None] * frequencies for f in [xx, yy]], dim=1)
        self.register_buffer("positions", torch.cat([pos.sin(), pos.cos()], dim=1)[None])
        self.blocks = nn.ModuleList([PredictorBlock(width, heads=6 if width % 6 == 0 else 4) for _ in range(depth)])
        self.norm, self.output_proj = nn.LayerNorm(width), nn.Linear(width, encoder_dim)

    def forward(self, context, visible, targets):
        context = self.input_proj(context) + self.positions[:, visible]
        queries = self.mask_token.expand(context.shape[0], len(targets), -1) + self.positions[:, targets]
        x = torch.cat([context, queries], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(self.norm(x[:, len(visible):]))


class JEPA(nn.Module):
    def __init__(self, encoder, predictor):
        super().__init__()
        self.encoder, self.predictor = encoder, predictor
        self.teacher = copy.deepcopy(encoder).requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, image, visible, targets):
        with torch.no_grad():
            latent = self.teacher(image)[:, targets]
            target = F.layer_norm(latent.float(), (latent.shape[-1],))
        predicted = self.predictor(self.encoder(image, visible), visible, targets)
        loss = F.smooth_l1_loss(predicted.float(), target)
        # Variation across examples/locations, not merely within each feature vector.
        std = target.flatten(0, 1).std(dim=0).mean()
        return loss, std

    @torch.no_grad()
    def update_teacher(self, momentum):
        source = dict(self.encoder.named_parameters())
        for name, target in self.teacher.named_parameters():
            target.lerp_(source[name].detach().to(target.dtype), 1-momentum)
        source_buffers = dict(self.encoder.named_buffers())
        for name, target in self.teacher.named_buffers():
            target.copy_(source_buffers[name])


def initialize_jepa_from_export(encoder, predictor, payload, model_name, last_blocks, tiny=False):
    """Start a new SSL schedule from an audited EMA/predictor export, including at a new grid."""
    signature = payload['signature']
    if payload.get('step', 0) < 1:
        raise ValueError('Warm-start export contains no completed JEPA training.')
    expected = (model_name, last_blocks, tiny, predictor.input_proj.out_features, len(predictor.blocks))
    actual = (signature['model'], signature['last_blocks'], signature['tiny'],
              signature['predictor_width'], signature['predictor_depth'])
    if actual != expected:
        raise ValueError('Warm-start JEPA architecture differs from this run.')
    encoder.load_state_dict(payload['encoder'], strict=True)
    state = dict(payload['predictor'])
    state['positions'] = predictor.positions  # fixed geometry; learned weights preserved
    predictor.load_state_dict(state, strict=True)


class GatedFusion(nn.Module):
    """Token-sequence GDConv-style fusion; no claim of physical cross-tile alignment."""
    def __init__(self, dim):
        super().__init__()
        self.value = nn.Conv1d(dim, dim, 5, padding=2, groups=dim)
        self.gate = nn.Conv1d(dim, dim, 5, padding=2, groups=dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        z = self.norm(x).transpose(1, 2)
        return x + (self.value(z) * self.gate(z).sigmoid()).transpose(1, 2)


class BiomassModel(nn.Module):
    def __init__(self, encoder, output_space="grams"):
        super().__init__()
        self.encoder = encoder
        self.output_space = output_space
        width = 256 if encoder.dim >= 256 else 64
        self.projection = nn.Linear(encoder.dim, width)
        self.fusion = nn.Sequential(GatedFusion(width), GatedFusion(width))
        self.attention = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))
        self.readout = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Dropout(.1), nn.Linear(width, 3))

    @torch.no_grad()
    def initialize_from_training_targets(self, targets):
        values = targets.clone() if isinstance(targets, torch.Tensor) else torch.tensor(targets.tolist(), dtype=torch.float32)
        median = values.float().median(dim=0).values.clamp_min(.01)
        value = torch.log1p(median) if self.output_space == "log1p" else median
        self.readout[-1].weight.zero_()
        self.readout[-1].bias.copy_(value + torch.log(-torch.expm1(-value)))

    def forward(self, tiles):
        batch, views, channels, height, width = tiles.shape
        features = self.encoder(tiles.reshape(batch*views, channels, height, width))
        features = features.reshape(batch, -1, features.shape[-1])
        features = self.fusion(self.projection(features))
        attention = self.attention(features).float().softmax(dim=1).to(features.dtype)
        pooled = (attention * features).sum(dim=1)
        primitive = F.softplus(self.readout(pooled).float())
        if self.output_space == "log1p":
            primitive = torch.expm1(primitive)
        green, dead, clover = primitive.unbind(dim=1)
        return torch.stack([green, dead, clover, green+clover, green+clover+dead], dim=1)


class DinoMLPBiomassModel(BiomassModel):
    """Plain supervised CLS+mean-patch DINO readout with the same mass/aux objectives."""
    def __init__(self, encoder, output_space='log1p', hidden=384, auxiliary_bins=0,
                 head_dropout=.2, regression_hidden=None, independent_targets=False):
        nn.Module.__init__(self)
        self.encoder, self.output_space = encoder, output_space
        self.independent_targets = independent_targets
        feature_dim = 2*encoder.dim
        self.attention = nn.Linear(feature_dim, 1)
        widths = list(regression_hidden) if regression_hidden is not None else [hidden]
        if not widths or any(not isinstance(w,int) or isinstance(w,bool) or w<1 for w in widths):
            raise ValueError('MLP hidden widths must be positive integers.')
        layers = [nn.LayerNorm(feature_dim)]
        previous = feature_dim
        for width in widths:
            layers += [nn.Linear(previous,width),nn.GELU(),nn.Dropout(head_dropout)]
            previous = width
        layers.append(nn.Linear(previous,5 if independent_targets else 3))
        self.readout = nn.Sequential(*layers)
        self.auxiliary_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(feature_dim),nn.Linear(feature_dim,hidden//2),
                          nn.GELU(),nn.Dropout(head_dropout),nn.Linear(hidden//2,auxiliary_bins))
            for _ in range(5)]) if auxiliary_bins else nn.ModuleList()

    def forward(self, tiles, return_auxiliary=False, detach_encoder=False):
        batch,views,channels,height,width = tiles.shape
        images = tiles.reshape(batch*views,channels,height,width)
        if detach_encoder:
            with torch.no_grad(): features = self.encoder(images,return_summary=True)
        else:
            features = self.encoder(images,return_summary=True)
        features = features.reshape(batch,views,-1)
        weights = self.attention(features).float().softmax(dim=1).to(features.dtype)
        pooled = (weights*features).sum(dim=1)
        primitive = F.softplus(self.readout(pooled).float())
        if self.output_space == 'log1p': primitive = torch.expm1(primitive)
        if self.independent_targets: mass = primitive
        else:
            green,dead,clover = primitive.unbind(dim=1)
            mass = torch.stack([green,dead,clover,green+clover,green+clover+dead],dim=1)
        if return_auxiliary:
            if not self.auxiliary_heads:raise ValueError('Auxiliary predictions require training heads.')
            return mass,torch.stack([head(pooled) for head in self.auxiliary_heads],dim=1)
        return mass


class PredictorBiomassModel(BiomassModel):
    """Regress from transferred context blocks or actual masked-patch latent predictions.

    The historical default is partial block transfer. ``masked_latent`` preserves
    the full SSL predictor, including queries and its output projection, and pools
    reconstructed patch latents for regression. It is a distinct, unproven ablation.
    """
    def __init__(self, encoder, predictor_width=384, predictor_depth=6, output_space="log1p",
                 pretrained_predictor=None, auxiliary_bins=0, predictor_dropout=0., head_dropout=.1,
                 regression_hidden=None, segmentation_classes=0, independent_targets=False,
                 predictor_readout="context", observed_latent_weight=0.):
        nn.Module.__init__(self)
        if predictor_readout not in {"context", "masked_latent"}:
            raise ValueError("Unknown predictor readout.")
        if predictor_readout == "masked_latent" and encoder.grid < 2:
            raise ValueError("Masked-latent readout needs at least two patch columns.")
        if not 0 <= observed_latent_weight < 1 or (observed_latent_weight and predictor_readout != 'masked_latent'):
            raise ValueError('Observed-latent weight must be in [0,1) and requires masked_latent readout.')
        self.encoder, self.output_space = encoder, output_space
        self.independent_targets = independent_targets
        self.predictor_readout = predictor_readout
        self.observed_latent_weight = observed_latent_weight
        predictor = Predictor(encoder.dim, encoder.grid, predictor_width, predictor_depth)
        if pretrained_predictor is not None:
            state = dict(pretrained_predictor)
            # Resolution-independent learned weights; regenerate fixed spatial positions.
            state["positions"] = predictor.positions
            predictor.load_state_dict(state, strict=True)
        for block in predictor.blocks:
            block.dropout.p = predictor_dropout
        if predictor_readout == "masked_latent":
            self.predictor = predictor
            feature_dim = encoder.dim
        else:
            self.projection = predictor.input_proj
            self.blocks = predictor.blocks
            self.norm = predictor.norm
            self.register_buffer("positions", predictor.positions)
            feature_dim = predictor_width
        self.attention = nn.Linear(feature_dim, 1)
        hidden = list(regression_hidden) if regression_hidden is not None else [predictor_width]
        if not hidden or any(not isinstance(width, int) or isinstance(width, bool) or width < 1 for width in hidden):
            raise ValueError("Regression hidden widths must be positive integers.")
        readout = [nn.LayerNorm(feature_dim)]
        previous = feature_dim
        for width in hidden:
            readout += [nn.Linear(previous, width), nn.GELU(), nn.Dropout(head_dropout)]
            previous = width
        readout.append(nn.Linear(previous, 5 if independent_targets else 3))
        self.readout = nn.Sequential(*readout)
        self.auxiliary_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, predictor_width//2),
                          nn.GELU(), nn.Dropout(head_dropout), nn.Linear(predictor_width//2, auxiliary_bins))
            for _ in range(5)]) if auxiliary_bins else nn.ModuleList()
        self.segmentation_head = (nn.Sequential(nn.LayerNorm(feature_dim),
                                  nn.Linear(feature_dim, segmentation_classes))
                                  if segmentation_classes else None)

    def forward(self, tiles, return_auxiliary=False, detach_encoder=False, return_segmentation=False):
        batch, views, channels, height, width = tiles.shape
        images = tiles.reshape(batch*views, channels, height, width)
        if self.predictor_readout == "masked_latent":
            predicted, target_indices = [], []
            observed, visible_indices = [], []
            for visible, targets in complementary_patch_masks(self.encoder.grid, images.device):
                if detach_encoder:
                    with torch.no_grad():
                        context = self.encoder(images, visible)
                else:
                    context = self.encoder(images, visible)
                predicted.append(self.predictor(context, visible, targets))
                target_indices.append(targets)
                if self.observed_latent_weight:
                    # SSL predicts layer-normalized teacher features. Keep the
                    # observed stream in that space without another encoder pass.
                    observed.append(F.layer_norm(context.float(), (context.shape[-1],)))
                    visible_indices.append(visible)
            # Restore row-major patch geometry before pooling or semantic supervision.
            order = torch.cat(target_indices).argsort()
            token_features = torch.cat(predicted, dim=1)[:, order]
            if self.observed_latent_weight:
                visible_order = torch.cat(visible_indices).argsort()
                observed_features = torch.cat(observed, dim=1)[:, visible_order].to(token_features.dtype)
                # Combine streams AFTER masked prediction. Queries still cannot
                # access their hidden pixels; regression can use observed pixels.
                alpha = self.observed_latent_weight
                token_features = (1-alpha)*token_features + alpha*observed_features
        else:
            if detach_encoder:
                # Preserve trainable flags/checkpoint keys during head warm-up.
                with torch.no_grad():
                    features = self.encoder(images)
            else:
                features = self.encoder(images)
            features = self.projection(features) + self.positions
            for block in self.blocks:
                features = block(features)
            token_features = self.norm(features)
        features = token_features.mean(dim=1).reshape(batch, views, -1)
        weights = self.attention(features).float().softmax(dim=1).to(features.dtype)
        pooled = (weights * features).sum(dim=1)
        primitive = F.softplus(self.readout(pooled).float())
        if self.output_space == "log1p":
            primitive = torch.expm1(primitive)
        if self.independent_targets:
            mass = primitive
        else:
            green, dead, clover = primitive.unbind(dim=1)
            mass = torch.stack([green, dead, clover, green+clover, green+clover+dead], dim=1)
        if return_segmentation:
            if self.segmentation_head is None:
                raise ValueError('Semantic supervision requested without a segmentation head.')
            semantic = self.segmentation_head(token_features).reshape(batch, views, -1,
                                                                        self.segmentation_head[-1].out_features)
            logits = torch.stack([head(pooled) for head in self.auxiliary_heads], dim=1) if self.auxiliary_heads else None
            return mass, logits, semantic
        if return_auxiliary:
            if not self.auxiliary_heads:
                raise ValueError("Auxiliary predictions require configured training heads.")
            return mass, torch.stack([head(pooled) for head in self.auxiliary_heads], dim=1)
        return mass


def configure_fp8(module, precision, device):
    if precision == "bf16":
        return {"requested": precision, "active": "bf16", "float8_layers": 0}
    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 9:
        raise RuntimeError("FP8 requested: requires Hopper/Blackwell CUDA GPU. Use --precision bf16 explicitly for CPU/other GPU.")
    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        def eligible(layer, name):
            return (isinstance(layer, nn.Linear) and "teacher" not in name
                    and min(layer.in_features, layer.out_features) >= 128
                    and layer.in_features % 16 == 0 and layer.out_features % 16 == 0)
        count = sum(eligible(layer, name) for name, layer in module.named_modules())
        convert_to_float8_training(module, module_filter_fn=eligible,
                                  config=Float8LinearConfig(pad_inner_dim=True))
    except (ImportError, AttributeError) as error:
        raise RuntimeError("FP8 setup failed. Install a torchao version compatible with your existing torch, or explicitly select bf16.") from error
    if not count:
        raise RuntimeError("No eligible FP8 layers were converted.")
    return {"requested": precision, "active": "fp8_linear_bf16_other", "float8_layers": count,
            "teacher": "bf16 autocast; FP32 master weights"}
