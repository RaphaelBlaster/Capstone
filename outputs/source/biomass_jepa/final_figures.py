"""Evidence-only report for the locked, two-fold dual study.

Consumes saved predictions/logs; never manufactures missing results or curves.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .data import TARGETS, file_sha256
from .final_metrics import global_r2, per_target, paired_bootstrap
from .metrics import scores
from .within_image_jigsaw import native_views, permutation_bank, apply_permutation

ARMS = {'jepa': ['control', 'shuffle'], 'dinov3': ['D0', 'D1', 'D2']}
FIGURES = ['study_design', 'jigsaw_native_and_224', 'jepa_latent_training',
           'jepa_predictions', 'dino_scale_discrepancy', 'main_comparisons',
           'per_target_residuals']


def _csv(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.csv.tmp')
    data.to_csv(temp, index=False, float_format='%.17g'); temp.replace(path)


def _save(fig, directory, name):
    fig.tight_layout(); fig.savefig(directory / (name + '.png'), dpi=180)
    plt.close(fig)


def _load_arm(root, frame, branch, arm):
    pieces, logs, ssl, compute = [], [], [], []
    for fold in (0, 1):
        directory = root / branch / f'fold_{fold}' / arm
        metrics = json.loads((directory / 'metrics.json').read_text())
        if not (metrics.get('complete') is True and metrics.get('completed_epochs') == 200
                and metrics.get('swa_count') == 40):
            raise ValueError(f'{branch}/{arm}/{fold} lacks the fixed completed endpoint.')
        checkpoint = directory / 'head_selected.pt'
        if file_sha256(checkpoint) != metrics.get('head_sha256'):
            raise ValueError('Selected head checkpoint hash mismatch.')
        rows = pd.read_csv(directory / 'val_predictions.csv', dtype={'Sampling_Date': str})
        expected = frame[frame.fold == fold].sort_values('image_path').reset_index(drop=True)
        rows = rows.sort_values('image_path').reset_index(drop=True)
        columns = ['image_path', 'Sampling_Date', 'fold']
        if rows.image_path.duplicated().any() or not rows[columns].equals(expected[columns]):
            raise ValueError('Validation IDs, acquisition dates or folds do not match originals.')
        truth = rows[TARGETS].to_numpy(np.float64)
        prediction = rows[['pred_' + t for t in TARGETS]].to_numpy(np.float64)
        if not np.allclose(truth, expected[TARGETS].to_numpy(np.float64), rtol=0, atol=1e-10):
            raise ValueError('Original labels changed in prediction CSV.')
        if not np.isfinite(prediction).all() or (prediction < 0).any():
            raise ValueError('Finite nonnegative predictions required.')
        fold_score = global_r2(truth, prediction)
        if abs(fold_score - metrics['global_weighted_r2_raw']) > 1e-10:
            raise ValueError('Saved fold metric does not match independently rescored CSV.')
        history = pd.read_csv(directory / 'head_training.csv')
        if history.epoch.tolist() != list(range(1, 201)) or not np.isfinite(history.loss).all():
            raise ValueError('Complete actual 200-epoch head logs required.')
        if int(history.steps.iloc[-1]) != metrics.get('optimizer_steps'):
            raise ValueError('Completed head optimizer steps disagree with actual logs.')
        logs.append(history.assign(branch=branch, arm=arm, fold=fold))
        if branch == 'jepa':
            latent = pd.read_csv(directory / 'ssl_training.csv')
            updates = 385 if fold == 0 else 377
            if latent.step.tolist() != list(range(1, updates + 1)) or not np.isfinite(latent.loss).all():
                raise ValueError('Full predictor specialization schedule logs required.')
            ssl.append(latent.assign(arm=arm, fold=fold, stage='new_predictor_specialization'))
            compute.append({'branch': branch, 'arm': arm, 'fold': fold,
                            'stage': 'predictor_specialization', 'optimizer_steps': updates,
                            'elapsed_seconds': float(latent.elapsed_seconds.iloc[-1]),
                            'training_images': len(frame[frame.fold != fold]),
                            'validation_images': 0})
        record = {'branch': branch, 'arm': arm, 'fold': fold, 'stage': 'regression_head',
                  'training_images': len(frame[frame.fold != fold]), 'validation_images': len(rows),
                  'checkpoint_sha256': metrics['head_sha256']}
        for key in ('parameters', 'elapsed_seconds', 'optimizer_steps', 'precision', 'gpu_peak_memory_bytes'):
            value = metrics.get(key)
            record[key] = json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
        compute.append(record); pieces.append(rows)
    return pd.concat(pieces).sort_values('image_path').reset_index(drop=True), logs, ssl, compute


def report(root, frame):
    """Render only after all five arms have complete, aligned real saved outputs."""
    root = Path(root); frame = frame.copy(); frame['Sampling_Date'] = frame.Sampling_Date.astype(str)
    expected = frame[frame.fold.isin([0, 1])].sort_values('image_path').reset_index(drop=True)
    if (len(frame) != 357 or frame.image_path.duplicated().any() or len(expected) != 105
            or expected.Sampling_Date.nunique() != 9
            or frame.groupby('Sampling_Date').fold.nunique().max() != 1
            or len(frame[frame.fold == 0]) != 49 or len(frame[frame.fold == 1]) != 56):
        raise ValueError('Locked 357 originals,105 held-out images,9 date groups and49/56 folds required.')
    arms, headlogs, latents, compute = {}, [], [], []
    # Validate everything before rendering or writing final report evidence.
    for branch, names in ARMS.items():
        for arm in names:
            rows, logs, ssl, records = _load_arm(root, frame, branch, arm)
            arms[(branch, arm)] = rows; headlogs.extend(logs); latents.extend(ssl); compute.extend(records)
    d2 = arms[('dinov3', 'D2')]
    coarse_columns = ['coarse_' + t for t in TARGETS]
    if not set(coarse_columns).issubset(d2) or not np.isfinite(d2[coarse_columns].to_numpy()).all():
        raise ValueError('D2 actual coarse-scale validation predictions required.')
    source = frame.sort_values('image_path').iloc[0]
    if file_sha256(source.absolute_path) != source.image_sha256:
        raise ValueError('Example original photograph hash changed.')
    views = native_views(source.absolute_path)
    figure_data = root / 'figure_data'; figures = root / 'figures'; stats = root / 'stats'
    for directory in (figure_data, figures, stats): directory.mkdir(parents=True, exist_ok=True)
    main, target_rows = [], []
    truth = expected[TARGETS].to_numpy(np.float64)
    for (branch, arm), rows in arms.items():
        pred = rows[['pred_' + t for t in TARGETS]].to_numpy(np.float64)
        score = global_r2(truth, pred); parity = abs(score - scores(truth, pred)['global_weighted_r2_raw'])
        if parity > 1e-10: raise ValueError('Independent pooled metric parity failed.')
        main.append({'branch': branch, 'arm': arm, 'global_weighted_r2_raw': score,
                     'images': 105, 'date_groups': 9, 'folds': '0,1', 'seeds': 1,
                     'metric_parity_error': parity, 'scope': 'reused_development'})
        errors = per_target(truth, pred)
        for j, target in enumerate(TARGETS):
            target_rows.append({'branch': branch, 'arm': arm, 'target': target,
                                **{key: values[j] for key, values in errors.items()}})
        _csv(figure_data / f'{branch}_{arm}_predictions.csv', rows)
    paired = []
    for branch, control, treatment in [('jepa', 'control', 'shuffle'), ('dinov3', 'D0', 'D1'), ('dinov3', 'D1', 'D2')]:
        p0 = arms[(branch, control)][['pred_' + t for t in TARGETS]].to_numpy(np.float64)
        p1 = arms[(branch, treatment)][['pred_' + t for t in TARGETS]].to_numpy(np.float64)
        interval = paired_bootstrap(truth, p0, p1, expected.Sampling_Date.to_numpy(), draws=10000, seed=42)
        paired.append({'branch': branch, 'comparison': treatment + '-' + control,
                       'difference': interval['difference'], 'lower95': interval['ci95'][0],
                       'upper95': interval['ci95'][1], **{key: value for key, value in interval.items()
                       if key not in ('difference', 'ci95')}})
    for branch, names in ARMS.items():
        combined = expected[['image_path', 'Sampling_Date', 'fold'] + TARGETS].copy()
        for arm in names:
            for target in TARGETS: combined[arm + '_pred_' + target] = arms[(branch, arm)]['pred_' + target]
        _csv(root / branch / 'paired_oof.csv', combined)
    _csv(stats / 'main_results.csv', pd.DataFrame(main)); _csv(stats / 'per_target.csv', pd.DataFrame(target_rows))
    _csv(stats / 'paired_bootstrap.csv', pd.DataFrame(paired)); _csv(stats / 'compute.csv', pd.DataFrame(compute))
    _csv(figure_data / 'head_training.csv', pd.concat(headlogs, ignore_index=True))
    _csv(figure_data / 'jepa_latent_training.csv', pd.concat(latents, ignore_index=True))
    # Method diagram depicts the locked algorithm, never reconstructed measurements.
    fig, ax = plt.subplots(figsize=(12, 4)); ax.axis('off')
    ax.text(.02, .75, 'I-JEPA: reused grass-adapted frozen student + EMA teacher\n'
            'Original / native10x10 jigsaws → masked predictor specialization\n'
            'Frozen specialized predictor → four semantic target blocks → biomass head', va='center')
    ax.text(.02, .25, 'DINOv3 B/16: frozen512px backbone → spatial tokens →8x8 grid\n'
            'D0 global mean | D1 nonlinear area mean | D2 + coarse4x4 consistency\n'
            'One whole-image biomass label; inference on original photographs', va='center')
    ax.set_title('Method schematic — no image reconstruction or local biomass ground truth')
    _save(fig, figures, 'study_design')
    fig, axes = plt.subplots(2, 3, figsize=(10, 7)); manifests = []
    for view, tile in enumerate(views):
        order = permutation_bank(source.image_sha256, view, 42)[0]
        shuffled = apply_permutation(tile, order)
        for column, (image, label) in enumerate([(tile, 'Original native view'),
                (shuffled, 'Artificial10x10 jigsaw'), (shuffled.resize((224, 224), Image.Resampling.BICUBIC), 'Artificial jigsaw at224px')]):
            axes[view, column].imshow(image); axes[view, column].set_title(label); axes[view, column].axis('off')
        manifests.append({'image_path': source.image_path, 'image_sha256': source.image_sha256,
                          'view': view, 'variant': 0, 'master_seed': 42,
                          'permutation': json.dumps(order), 'artificial_augmentation': True})
    _csv(figure_data / 'jigsaw_panel_manifest.csv', pd.DataFrame(manifests))
    _save(fig, figures, 'jigsaw_native_and_224')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for log in latents:
        axes[0].plot(log.step, log.loss, label=f'{log.arm.iloc[0]} fold{log.fold.iloc[0]}')
    axes[0].set(xlabel='Actual predictor optimizer step', ylabel='Latent SmoothL1 loss')
    historical = root / 'jepa' / 'historical_ssl_training.csv'
    if historical.exists():
        past = pd.read_csv(historical); _csv(figure_data / 'historical_ssl_training.csv', past)
        axes[0].plot(past.step, past.loss, ':', label='Historical grass SSL — different stage')
    for log in headlogs:
        axes[1].plot(log.epoch, log.loss, label=f'{log.arm.iloc[0]} fold{log.fold.iloc[0]}')
    axes[1].set(xlabel='Actual supervised head epoch', ylabel='Normalized raw weighted MSE')
    for ax in axes: ax.legend(fontsize=6)
    _save(fig, figures, 'jepa_latent_training')
    fig, axes = plt.subplots(1, 5, figsize=(17, 4))
    for j, target in enumerate(TARGETS):
        for arm in ARMS['jepa']:
            axes[j].scatter(truth[:, j], arms[('jepa', arm)]['pred_' + target], s=8, alpha=.65, label=arm)
        bound = max(float(truth[:, j].max()), *(float(arms[('jepa', arm)]['pred_' + target].max()) for arm in ARMS['jepa']))
        axes[j].plot([0, bound], [0, bound], 'k:', linewidth=1)
        axes[j].set(title=target, xlabel='Actual grams', ylabel='Predicted grams')
    axes[0].legend(); _save(fig, figures, 'jepa_predictions')
    scale_rows = d2[['image_path', 'Sampling_Date', 'fold'] + TARGETS + ['pred_' + t for t in TARGETS] + coarse_columns].copy()
    for target in TARGETS: scale_rows['fine_minus_coarse_' + target] = d2['pred_' + target] - d2['coarse_' + target]
    _csv(figure_data / 'dino_scale_discrepancy.csv', scale_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(d2.pred_Dry_Total_g, d2.coarse_Dry_Total_g, s=12)
    axes[0].set(xlabel='Fine8x8 Total prediction (g)', ylabel='Coarse4x4 Total prediction (g)')
    axes[1].scatter(truth[:, 4], scale_rows.fine_minus_coarse_Dry_Total_g, s=12)
    axes[1].axhline(0, color='k', linestyle=':'); axes[1].set(xlabel='Actual Total (g)', ylabel='Fine-minus-coarse Total (g)')
    _save(fig, figures, 'dino_scale_discrepancy')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4)); result = pd.DataFrame(main)
    axes[0].bar(range(len(main)), result.global_weighted_r2_raw)
    axes[0].set_xticks(range(len(main)), [row['branch'] + '/' + row['arm'] for row in main], rotation=25)
    axes[0].set_ylabel('Pooled global weighted R² — development')
    for i, item in enumerate(paired):
        axes[1].plot([item['lower95'], item['upper95']], [i, i], color='C0')
        axes[1].scatter(item['difference'], i, color='C0')
    axes[1].axvline(0, color='k', linestyle=':'); axes[1].set_yticks(range(3), [r['comparison'] for r in paired])
    axes[1].set_xlabel('Paired R² difference, date-group95% interval')
    _save(fig, figures, 'main_comparisons')
    fig, axes = plt.subplots(1, 5, figsize=(18, 4)); residuals = []
    for j, target in enumerate(TARGETS):
        for (branch, arm), rows in arms.items():
            residual = rows['pred_' + target].to_numpy() - truth[:, j]
            axes[j].scatter(truth[:, 4], residual, s=7, alpha=.45, label=f'{branch}/{arm}')
            residuals.extend({'image_path': path, 'branch': branch, 'arm': arm, 'target': target,
                'actual_total_g': total, 'residual_g': error} for path, total, error
                in zip(expected.image_path, truth[:, 4], residual))
        axes[j].axhline(0, color='k', linestyle=':'); axes[j].set(title=target, xlabel='Actual Total grams', ylabel='Prediction-minus-actual grams')
    axes[0].legend(fontsize=6); _save(fig, figures, 'per_target_residuals')
    _csv(figure_data / 'residuals.csv', pd.DataFrame(residuals))
    text = ['# Dual study development results', '',
        'Contract DUAL-JEPA-DINO-2026-10-09-r1. All five arms completed both folds0/1,200 head epochs and final40 SWA.',
        'Evaluation uses105 of357 original photographs,9 acquisition-date groups,one seed, and reused development folds. These are neither full OOF nor competition scores.', '',
        pd.DataFrame(main)[['branch', 'arm', 'global_weighted_r2_raw']].to_csv(index=False),
        'Paired intervals use10000 fixed-seed date-group resamples and are conditional on these trained models. They exclude seed variation and do not establish unseen-condition generalization.',
        'J-shuffle versus J-control tests predictor specialization. D2 versus D1 tests the prespecified0.10 consistency penalty. JEPA versus DINO also changes resolution, representation, head and compute, so it does not isolate backbone superiority.',
        'The ten jigsaws are artificial augmentations, not additional independently labelled photographs. No local biomass ground truth or measured spatial density is inferred.',
        'Figures use saved observations and training logs. DINO local heatmaps are omitted unless actual contribution arrays are separately preserved; no heatmaps were invented.',
        'Independent P1 architectural and checkpoint verification remains required. Favorable development scores alone do not guarantee novelty or Scientific Reports acceptance.',
        'Cite Meta I-JEPA; Meta DINOv3; Arystanova et al.(2026), Agriculture16:1401, doi:10.3390/agriculture16131401. Dataset/model redistribution licenses require documentation before publication.']
    temporary = root / 'FINAL_REPORT.md.tmp'; temporary.write_text('\n\n'.join(text)); temporary.replace(root / 'FINAL_REPORT.md')
    return {'status': 'completed_reporting_pending_independent_P1', 'main_results': main,
            'paired': paired, 'figures': [str(figures / (name + '.png')) for name in FIGURES]}
