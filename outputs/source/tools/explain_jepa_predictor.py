"""Saved J3 input sensitivity on a preselected random held-out sample; no training."""
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from biomass_jepa.official_ijepa import OfficialEncoder, make_predictor
from biomass_jepa.official_heads import GatedRegressionHead
from biomass_jepa.semantic_predictor_features import semantic_masks, pool_predictions
from biomass_jepa.within_image_jigsaw import native_views, preprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/jepa_xai_3'
STUDY = ROOT / 'outputs/final_dual_study/full_results/final_dual_study/jepa'
TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def predict(encoder, predictor, head, views):
    """Exact semantic readout operations, with input differentiation enabled."""
    batch, count = views.shape[:2]
    images = views.reshape(batch * count, 3, 224, 224)
    context, targets = semantic_masks(batch * count, images.device)
    features = pool_predictions(predictor(encoder(images, context), context, targets), batch, count)
    return head(features)


def patch_values(values):
    return values.reshape(2, 16, 14, 16, 14).sum(axis=(2, 4))


def occlude(views, indices):
    result = views.detach().clone()
    for index in indices:
        view, cell = divmod(int(index), 256)
        row, col = divmod(cell, 16)
        result[0, view, :, row*14:(row+1)*14, col*14:(col+1)*14] = 0
    return result


def main():
    torch.set_num_threads(3)
    selection = json.loads((OUT / 'selection.json').read_text())
    source = ROOT / 'outputs/block_shuffle_preflight/verified_source_v2/adapted.pt'
    source_hash = digest(source)
    assert source_hash == '6d2b5d9c83423155b4a948822e846caaa9a6bc6be20e3fab13ca06b8e705f43d'
    print('Source hash verified. Loading frozen encoder, no EMA teacher.', flush=True)
    checkpoint = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    # The official constructor reads scalar stochastic-depth tensors, so it needs CPU.
    encoder = OfficialEncoder(224, 0)
    predictor = make_predictor(encoder)
    encoder.load_state_dict(checkpoint['encoder'], strict=True, assign=True)
    del checkpoint
    context, _ = semantic_masks(1, 'cpu')
    visible = np.zeros(256, dtype=bool)
    visible[context[0][0].numpy()] = True
    visible = np.tile(visible, 2)
    visible_pixels = np.repeat(np.repeat(visible.reshape(2, 16, 16), 14, axis=1), 14, axis=2)
    results = []
    for number, row in enumerate(selection['images'], 1):
        started = time.monotonic()
        image = OUT / 'images' / Path(row['image_path']).name
        assert digest(image) == row['image_sha256']
        folder = STUDY / ('fold_' + row['fold']) / 'shuffle'
        pred_path, head_path = folder / 'predictor_specialized.pt', folder / 'head_selected.pt'
        specialized = torch.load(pred_path, map_location='cpu', weights_only=True, mmap=True)
        predictor.load_state_dict(specialized['predictor'], strict=True, assign=True)
        saved = torch.load(head_path, map_location='cpu', weights_only=True)
        assert saved['signature']['fold'] == int(row['fold'])
        assert saved['signature']['arm'] == 'J_shuffle'
        assert saved['signature']['origin']['predictor_sha256'] == digest(pred_path)
        head = GatedRegressionHead(saved['dim'], saved['scale'], saved['median'])
        head.load_state_dict(saved['model'], strict=True)
        for model in (encoder, predictor, head):
            model.requires_grad_(False).eval()
        views = torch.stack([preprocess(tile, 224) for tile in native_views(image)])[None].requires_grad_(True)
        print(f'[{number}/3] {image.stem}: forward and input gradients', flush=True)
        prediction = predict(encoder, predictor, head, views)
        gradients = []
        # Four gradients; GDM gradient follows exactly from Green + Clover.
        for target in (0, 1, 2, 4):
            gradient = torch.autograd.grad(prediction[0, target], views, retain_graph=target != 4)[0]
            gradients.append(gradient.detach()[0].numpy())
        arrays = dict(zip(('green', 'dead', 'clover', 'total'), gradients))
        arrays['gdm'] = arrays['green'] + arrays['clover']
        heatmaps = {key: np.abs(value * views.detach()[0].numpy()).sum(1) for key, value in arrays.items()}
        hidden_grad_max = max(float(np.abs(value[:, :, ~visible_pixels[0]]).max()) for value in gradients)
        assert hidden_grad_max == 0, 'Hidden patches must be excluded before encoder attention.'
        importance = patch_values(heatmaps['total']).reshape(-1)
        top = np.flatnonzero(visible)[np.argsort(importance[visible])[-5:]]
        random_cells = np.random.default_rng(42 + number).choice(np.flatnonzero(visible), 5, replace=False)
        baseline = prediction.detach()[0].numpy()
        del prediction
        perturbations = {}
        print(f'[{number}/3] occlusion: top-five, random-five, hidden negative control', flush=True)
        with torch.no_grad():
            for label, cells in [('top_five_visible', top), ('random_five_visible', random_cells), ('all_hidden', np.flatnonzero(~visible))]:
                values = predict(encoder, predictor, head, occlude(views, cells))[0].numpy()
                perturbations[label] = {'patch_indices': cells.tolist(), 'prediction_g': values.tolist(), 'delta_g': (values-baseline).tolist()}
        assert max(abs(v) for v in perturbations['all_hidden']['delta_g']) < 1e-5
        saved_prediction = np.array([float(row['shuffle_pred_' + target]) for target in TARGETS])
        truth = np.array([float(row[target]) for target in TARGETS])
        result = {'image': row['image_path'], 'fold': int(row['fold']), 'truth_g': truth.tolist(),
                  'saved_fp8_prediction_g': saved_prediction.tolist(), 'cpu_fp32_prediction_g': baseline.tolist(),
                  'fp32_minus_saved_fp8_g': (baseline-saved_prediction).tolist(), 'hidden_input_gradient_max': hidden_grad_max,
                  'occlusion': perturbations, 'image_sha256': digest(image),
                  'predictor_sha256': digest(pred_path), 'head_sha256': digest(head_path), 'seconds': time.monotonic()-started}
        results.append(result)
        np.savez_compressed(OUT / (image.stem + '_attributions.npz'), **arrays,
                            **{'heatmap_' + key: value for key, value in heatmaps.items()}, visible=visible_pixels)
        photo = np.array(Image.open(image).resize((448, 224), Image.Resampling.BICUBIC))
        fig, axes = plt.subplots(2, 3, figsize=(13, 6))
        axes[0, 0].imshow(photo); axes[0, 0].set_title('Original complete photograph')
        axes[0, 1].imshow(photo)
        mask = np.concatenate(visible_pixels, axis=1)
        axes[0, 1].imshow(np.ma.masked_where(mask, np.zeros_like(mask)), cmap='gray', vmin=0, vmax=1, alpha=.7)
        axes[0, 1].set_title('Dark corners: hidden from student encoder')
        for axis, key in zip([axes[0, 2], *axes[1]], ['green', 'dead', 'clover', 'total']):
            values = np.concatenate(heatmaps[key], axis=1)
            scale = float(np.quantile(values[mask], .99))
            axis.imshow(photo)
            axis.imshow(np.clip(values/max(scale, 1e-12), 0, 1), cmap='inferno', alpha=.6, vmin=0, vmax=1)
            axis.set_title(key.title() + ': |gradient × normalized input|')
        for axis in axes.flat:
            axis.axis('off')
        fig.suptitle(f'{image.stem} — J3 fold {row["fold"]}, CPU FP32 sensitivity\nTotal: truth {truth[4]:.2f}g; saved FP8 {saved_prediction[4]:.2f}g; FP32 {baseline[4]:.2f}g')
        fig.tight_layout(); fig.savefig(OUT / (image.stem + '_explanation.png'), dpi=160); plt.close(fig)
        (OUT / 'results.json').write_text(json.dumps({'targets': TARGETS, 'source_sha256': source_hash, 'selection': selection, 'results': results,
            'method': 'Local input gradient × normalized input; magnitude independently normalized per target. Mean-RGB patch occlusion; no retraining.',
            'limitations': 'CPU FP32 explanations differ from original FP8 numerics. Three descriptive examples cannot establish the cause of aggregate error. Occlusion is an out-of-distribution sensitivity diagnostic, not measured local biomass.'}, indent=2))
        print(f'[{number}/3] saved; {result["seconds"]:.1f}s; max precision difference {float(np.abs(baseline-saved_prediction).max()):.4f}g', flush=True)


if __name__ == '__main__':
    main()
