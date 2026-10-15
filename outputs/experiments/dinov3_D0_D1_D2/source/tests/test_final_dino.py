"""Independent geometric/objective oracles; fake backbones are test-only."""
from types import SimpleNamespace
import sys

import pytest
import torch
from torch import nn

from biomass_jepa.dinov3_dense import (
    CONSISTENCY_WEIGHT, DINODensityHead, DINOGlobalHead, OFFICIAL_REPOSITORY,
    _sha256, coarse_grid, consistency_penalty, dense_grid, extract_dense,
    head_loss, load_dinov3, make_head, nondegeneracy_proof, patch_tokens,
)
from biomass_jepa import dinov3_dense as dense_module


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(2)
    torch.manual_seed(49)


def heterogeneous(batch=2):
    # Nonconstant content at every fine spatial cell makes pre-MLP pooling differ.
    return torch.randn(batch, 2, 8, 8, 768) * 2


def test_patch_tokens_exclude_cls_and_registers():
    patches = torch.randn(2, 1024, 768)
    assert patch_tokens({'x_norm_patchtokens': patches,
                         'x_norm_clstoken': torch.full((2, 768), -10000.),
                         'x_norm_regtokens': torch.full((2, 4, 768), 10000.)}, 2) is patches
    with pytest.raises(ValueError):
        patch_tokens({'tokens': torch.randn(2, 1029, 768)}, 2)
    with pytest.raises(ValueError):
        patch_tokens({'x_norm_patchtokens': torch.randn(2, 1029, 768)}, 2)
    patches[0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        patch_tokens({'x_norm_patchtokens': patches}, 2)


def test_spatial_pooling_independent_nested_loop_oracle():
    # Codes preserve image, view, row and column; every feature channel is equal.
    codes = torch.arange(4 * 1024).reshape(4, 32, 32).float()
    tokens = codes.reshape(4, 1024, 1).expand(-1, -1, 768)
    fine = dense_grid(tokens, 2)
    assert fine.shape == (2, 2, 8, 8, 768)
    for image in range(2):
        for view in range(2):
            for row in range(8):
                for col in range(8):
                    expected = codes[image * 2 + view, row * 4:row * 4 + 4,
                                     col * 4:col * 4 + 4].mean()
                    assert torch.all(fine[image, view, row, col] == expected)
    coarse = coarse_grid(fine)
    for image in range(2):
        for view in range(2):
            for row in range(4):
                for col in range(4):
                    expected = codes[image * 2 + view, row * 8:row * 8 + 8,
                                     col * 8:col * 8 + 8].mean()
                    assert torch.all(coarse[image, view, row, col] == expected)


def test_d1_d2_same_capacity_state_and_primary_inference():
    d1, d2 = make_head('D1', 12.), make_head('D2', 12.)
    d2.load_state_dict(d1.state_dict(), strict=True)
    features = heterogeneous()
    assert sum(p.numel() for p in d1.parameters()) == sum(p.numel() for p in d2.parameters())
    assert torch.equal(d1(features), d2(features))
    assert d1.state_dict().keys() == d2.state_dict().keys()
    assert isinstance(make_head('D0', 12), DINOGlobalHead)


def test_whole_image_mass_is_equal_area_mean_and_derived_sums():
    head = DINODensityHead(40)
    features = heterogeneous(3)
    local = head.local_contributions(features)
    prediction = head(features)
    assert local.shape == (3, 2, 8, 8, 3)
    assert prediction.shape == (3, 5)
    assert torch.all(local >= 0) and torch.all(prediction >= 0)
    # Independent explicit flattened 128-cell aggregation.
    oracle = local.reshape(3, 128, 3).sum(1) / 128
    torch.testing.assert_close(prediction[:, :3], oracle)
    torch.testing.assert_close(prediction[:, 3], oracle[:, 0] + oracle[:, 2])
    torch.testing.assert_close(prediction[:, 4], oracle.sum(1))
    with pytest.raises(ValueError):
        head_loss(head, features, torch.ones(3, 2, 8, 8, 5), 'D1')


def test_crossscale_penalty_nonzero_and_independent_gradient():
    head = DINODensityHead(10.)
    features = heterogeneous()
    fine, coarse = head.predictions_at_scales(features)
    assert not torch.allclose(fine, coarse)
    penalty = ((fine[:, :3] - coarse[:, :3]) ** 2).mean() / 100
    assert penalty.item() > 1e-8
    torch.testing.assert_close(consistency_penalty(fine, coarse, head.scale), penalty)
    gradients = torch.autograd.grad(penalty, tuple(head.parameters()))
    assert all(torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g).item() for g in gradients)
    proof = nondegeneracy_proof(head, features)
    assert proof['penalty'] > 1e-8 and proof['gradient_max_abs'] > 0


def test_regularizer_changes_only_d2_objective_with_locked_coefficient():
    head, features, labels = DINODensityHead(7.), heterogeneous(), torch.ones(2, 5)
    baseline, d1 = head_loss(head, features, labels, 'D1')
    regularized, d2 = head_loss(head, features, labels, 'D2')
    oracle = ((d1['pred_fine'] - labels) ** 2 * torch.tensor([.1, .1, .1, .2, .5])).sum(1).mean() / 49
    torch.testing.assert_close(baseline, oracle)
    assert CONSISTENCY_WEIGHT == .10
    torch.testing.assert_close(regularized, baseline + .10 * d2['consistency_loss'])
    assert torch.equal(d1['pred_fine'], d2['pred_fine'])


def test_constant_or_linear_identity_regularizer_rejected():
    head = DINODensityHead(1.)
    head.density_head = nn.Sequential(nn.Linear(768, 3), nn.Identity())
    with pytest.raises(ValueError, match='nondegeneracy'):
        nondegeneracy_proof(head, torch.zeros(2, 2, 8, 8, 768))


class FrozenTestBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.proof = nn.Parameter(torch.ones(()))
        self.embed_dim = 768
        self.n_storage_tokens = 4
        self.patch_embed = SimpleNamespace(patch_size=(16, 16))
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(12)])

    def forward_features(self, images):
        return {'x_norm_patchtokens': torch.arange(1024).reshape(1, 1024, 1).float()
                .expand(len(images), -1, 768) * self.proof}


def test_frozen_extraction_only_correct_input_and_eval():
    model = FrozenTestBackbone().requires_grad_(False).eval()
    output = extract_dense(model, torch.zeros(1, 2, 3, 512, 512))
    assert output.shape == (1, 2, 8, 8, 768) and not output.requires_grad
    with pytest.raises(ValueError):
        extract_dense(model, torch.zeros(1, 2, 3, 224, 224))
    model.train()
    with pytest.raises(ValueError):
        extract_dense(model, torch.zeros(1, 2, 3, 512, 512))


def test_local_source_checkpoint_hash_and_strict_load(tmp_path, monkeypatch):
    repository = tmp_path / 'official'; repository.mkdir()
    (repository / 'hubconf.py').write_text('# synthetic fixture only\n')
    (repository / 'LICENSE').write_text('test fixture\n')
    checkpoint = tmp_path / 'weights.pt'
    torch.save(FrozenTestBackbone().state_dict(), checkpoint)
    metadata = {'repository': OFFICIAL_REPOSITORY, 'revision': 'a' * 40,
                'license': 'TEST ONLY', 'checkpoint_sha256': _sha256(checkpoint),
                'source_files': {p.name: _sha256(p) for p in repository.iterdir()}}
    calls = []
    def load(repo):
        calls.append(repo)
        return FrozenTestBackbone()
    monkeypatch.setattr(dense_module, '_local_backbone', load)
    model, provenance = load_dinov3(repository, checkpoint, metadata)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert calls[0] == repository.resolve()
    assert provenance['strict_state_load']
    (repository / 'hubconf.py').write_text('# altered source\n')
    with pytest.raises(ValueError, match='source bytes'):
        load_dinov3(repository, checkpoint, metadata)
    (repository / 'hubconf.py').write_text('# synthetic fixture only\n')
    torch.save({'unknown': torch.zeros(1)}, checkpoint)
    with pytest.raises(ValueError, match='checkpoint SHA256'):
        load_dinov3(repository, checkpoint, metadata)
    metadata['checkpoint_sha256'] = _sha256(checkpoint)
    with pytest.raises(RuntimeError):
        load_dinov3(repository, checkpoint, metadata)


def test_direct_import_refuses_foreign_package_without_cleanup_of_other_owner(tmp_path, monkeypatch):
    foreign = SimpleNamespace(__file__=str(tmp_path / 'foreign' / '__init__.py'))
    monkeypatch.setitem(sys.modules, 'dinov3', foreign)
    before_path = list(sys.path)
    with pytest.raises(ValueError, match='foreign DINOv3'):
        dense_module._local_backbone((tmp_path / 'official').resolve())
    assert sys.modules['dinov3'] is foreign and sys.path == before_path


def test_direct_backbone_import_restores_path_and_avoids_hubconf(tmp_path):
    repository = tmp_path / 'official'
    package = repository / 'dinov3' / 'hub'; package.mkdir(parents=True)
    (repository / 'dinov3' / '__init__.py').write_text('')
    (package / '__init__.py').write_text('')
    (repository / 'hubconf.py').write_text('raise RuntimeError("Optional segmentors must not import")')
    (package / 'backbones.py').write_text(
        'def dinov3_vitb16(*, pretrained):\n'
        '    assert pretrained is False\n'
        '    return "LOCAL_BACKBONE_ONLY"\n')
    before_path = list(sys.path)
    before_modules = set(sys.modules)
    try:
        assert dense_module._local_backbone(repository.resolve()) == 'LOCAL_BACKBONE_ONLY'
        assert sys.path == before_path
    finally:
        for name in list(sys.modules):
            if name not in before_modules and (name == 'dinov3' or name.startswith('dinov3.')):
                del sys.modules[name]


def test_head_reload_preserves_fine_and_coarse_predictions(tmp_path):
    head, features = DINODensityHead(9.), heterogeneous()
    before = head.predictions_at_scales(features)
    path = tmp_path / 'head.pt'; torch.save(head.state_dict(), path)
    restored = DINODensityHead(9.)
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    after = restored.predictions_at_scales(features)
    assert all(torch.equal(a, b) for a, b in zip(before, after))


@pytest.mark.parametrize('scale', [0, -1, float('nan'), [1, 2]])
def test_invalid_scale_rejected(scale):
    with pytest.raises(ValueError):
        DINODensityHead(scale)
