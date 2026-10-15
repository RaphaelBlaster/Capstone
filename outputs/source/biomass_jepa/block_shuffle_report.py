"""Independent, paired analysis of the fixed three-fold block-permutation ablation.

No checkpoint is loaded and no GPU work is performed. Only completed, identity- and
label-matched fixed-endpoint predictions enter inference. Missing arms stay missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
WEIGHTS = np.array([.1, .1, .1, .2, .5], dtype=np.float64)
ARMS = ('control', 'shuffle_4', 'shuffle_10')
FOLDS = (0, 1, 2)
PREDICTED = ['pred_' + target for target in TARGETS]
LIMITS = (
    'Partial development only: reused folds 0/1/2, one training seed; not protected '
    'confirmation, full OOF, competition scores or a preregistered study. Paired '
    'date-group bootstrap conditions on fixed trained models and excludes training-seed '
    'uncertainty and prior model-selection bias. Dates are proxies, not verified sites. '
    'Synthetic permutations add no independent biological observations. '
    'SSL latent loss cannot establish downstream benefit from more SSL steps.'
)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _read_json(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else {}


def raw_weighted_r2(truth, prediction):
    """ONE global mean across pooled rows and targets; never average fold R²."""
    y, p = np.asarray(truth, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != 5 or y.shape != p.shape or not len(y):
        raise ValueError('Expected matching nonempty N x 5 target matrices.')
    if not np.isfinite(y).all() or not np.isfinite(p).all() or (y < 0).any() or (p < 0).any():
        raise ValueError('Finite nonnegative targets and predictions required.')
    if np.ptp(y) == 0:
        return None
    mean = np.sum(y * WEIGHTS) / (len(y) * WEIGHTS.sum())
    denominator = np.sum(WEIGHTS * np.square(y - mean))
    return float(1 - np.sum(WEIGHTS * np.square(y - p)) / denominator) if denominator > 0 else None


def paired_bootstrap(truth, predictions, groups, draws=10000, seed=42):
    """Sample date groups once per draw, preserving the same rows in every arm."""
    if draws < 1:
        raise ValueError('draws must be positive.')
    y, groups = np.asarray(truth, dtype=np.float64), np.asarray(groups, dtype=str)
    if len(groups) != len(y) or not len(groups):
        raise ValueError('One date group is required for every row.')
    observed = {arm: raw_weighted_r2(y, p) for arm, p in predictions.items()}
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(draws):
        sample = np.concatenate([indices[g] for g in rng.choice(unique, len(unique), replace=True)])
        rows.append({'draw': draw, **{arm: raw_weighted_r2(y[sample], np.asarray(p)[sample])
                                     for arm, p in predictions.items()}})
    samples = pd.DataFrame(rows)
    for arm in predictions:
        samples[arm] = pd.to_numeric(samples[arm], errors='raise').astype(float)
    comparisons = {}
    for treatment, reference in [('shuffle_4', 'control'), ('shuffle_10', 'control'), ('shuffle_4', 'shuffle_10')]:
        if treatment not in predictions or reference not in predictions:
            continue
        key = treatment + '_minus_' + reference
        values = samples[treatment] - samples[reference]
        samples[key] = values
        finite = values[np.isfinite(values)]
        valid = len(unique) >= 2 and len(finite) > 0
        interval = np.quantile(finite, [.025, .975]).tolist() if valid else None
        effect = observed[treatment] - observed[reference] if all(observed[a] is not None for a in (treatment, reference)) else None
        comparisons[key] = dict(observed_difference=effect, ci95=interval,
            independent_date_groups=len(unique), draws=draws, valid_draws=len(finite), seed=seed,
            ci_supports_positive=bool(interval and interval[0] > 0),
            interpretation=('Positive paired difference supported under this fixed development protocol.'
                            if interval and interval[0] > 0 else 'Positive improvement is not established.'),
            reason=None if valid else 'Fewer than two groups or undefined bootstrap denominators.',
            scope=LIMITS)
    return samples, comparisons


def _validate_prepared(path):
    frame = pd.read_csv(path, dtype={'group_id': str, 'Sampling_Date': str, 'image_sha256': str})
    required = {'image_path', 'image_sha256', 'group_id', 'fold', *TARGETS}
    if not required.issubset(frame) or frame[list(required)].isna().any().any():
        raise ValueError('Prepared cohort lacks required identities, groups, folds or labels.')
    if frame.image_path.duplicated().any() or frame.image_sha256.duplicated().any():
        raise ValueError('Duplicate original image in prepared cohort.')
    if frame.groupby('group_id').fold.nunique().max() != 1:
        raise ValueError('Acquisition groups cross folds.')
    if 'Sampling_Date' in frame:
        if not frame.Sampling_Date.equals(frame.group_id):
            raise ValueError('group_id must equal the original Sampling_Date proxy.')
        if frame.groupby('Sampling_Date').fold.nunique().max() != 1:
            raise ValueError('Sampling dates cross folds.')
    y = frame[TARGETS].to_numpy(float)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Invalid original labels.')
    return frame.sort_values('image_path').reset_index(drop=True)


def _validate_predictions(pred, frame):
    required = ['image_path', 'group_id', 'fold'] + TARGETS + PREDICTED
    if not set(required).issubset(pred) or pred[required].isna().any().any():
        raise ValueError('Prediction rows have missing identities, labels or values.')
    if pred.image_path.duplicated().any():
        raise ValueError('Duplicate prediction image.')
    pred = pred.sort_values('image_path').reset_index(drop=True)
    expected = frame.set_index('image_path')
    if not set(pred.image_path).issubset(expected.index):
        raise ValueError('Prediction contains an unknown original image.')
    matched = expected.loc[pred.image_path].reset_index()
    for column in ['group_id', 'fold']:
        if not np.array_equal(pred[column].to_numpy(), matched[column].to_numpy()):
            raise ValueError('Prediction groups/folds differ from prepared original images.')
    if not np.allclose(pred[TARGETS].to_numpy(float), matched[TARGETS].to_numpy(float), rtol=0, atol=1e-9):
        raise ValueError('Prediction labels differ from unchanged original labels.')
    p = pred[PREDICTED].to_numpy(float)
    if not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError('Finite positive predictions required from the positive-primitive head.')
    if (not np.allclose(p[:, 3], p[:, 0] + p[:, 2], rtol=1e-5, atol=1e-5)
            or not np.allclose(p[:, 4], p[:, 3] + p[:, 1], rtol=1e-5, atol=1e-5)):
        raise ValueError('Prediction target identity fails: GDM=Green+Clover and Total=GDM+Dead.')
    if not pred.fold.isin(FOLDS).all():
        raise ValueError('Only authorized folds 0/1/2 may enter this experiment.')
    return pred


def _check_saved_score(metrics, pred):
    if 'global_weighted_r2_raw' not in metrics:
        return
    observed = raw_weighted_r2(pred[TARGETS], pred[PREDICTED])
    reported = metrics['global_weighted_r2_raw']
    if observed is None and reported is None:
        return
    if observed is None or reported is None or not np.isfinite(reported) or not np.isclose(observed, reported, rtol=0, atol=1e-8):
        raise ValueError('Saved global raw weighted score differs from independent recalculation.')


def _load_arm(folder, frame):
    metrics = _read_json(folder / 'metrics.json')
    path = folder / 'oof_predictions.csv'
    if not path.is_file():
        # Preserve interrupted fold outputs for audit, never present them as a C result.
        parts = [pd.read_csv(p, dtype={'group_id': str}) for p in sorted(folder.glob('fold_*/predictions.csv'))]
        if parts:
            _validate_predictions(pd.concat(parts), frame)
        return None, dict(status='incomplete', reason='No completed pooled predictions.', saved_fold_rows=sum(map(len, parts)))
    pred = _validate_predictions(pd.read_csv(path, dtype={'group_id': str}), frame)
    _check_saved_score(metrics, pred)
    expected = frame[frame.fold.isin(FOLDS)].sort_values('image_path')
    completed = metrics.get('status') == 'completed_heads'
    if metrics.get('folds') != list(FOLDS):
        completed = False
    for fold in FOLDS:
        fdir = folder / f'fold_{fold}'
        fm = _read_json(fdir / 'metrics.json')
        if fm.get('status') != 'completed_fixed_endpoint' or fm.get('completed_epochs') != 200:
            completed = False
        fold_file = fdir / 'predictions.csv'
        if not fold_file.is_file():
            completed = False
        else:
            fold_pred = _validate_predictions(pd.read_csv(fold_file, dtype={'group_id': str}), frame)
            _check_saved_score(fm, fold_pred)
            pooled_fold = pred[pred.fold == fold].reset_index(drop=True)
            if not fold_pred[(['image_path', 'group_id', 'fold'] + TARGETS + PREDICTED)].equals(pooled_fold[(['image_path', 'group_id', 'fold'] + TARGETS + PREDICTED)]):
                raise ValueError('Pooled predictions differ from saved fold predictions.')
    if not np.array_equal(pred.image_path.to_numpy(), expected.image_path.to_numpy()):
        if completed:
            raise ValueError('Completed arm does not cover the exact authorized cohort once.')
        return None, dict(status='incomplete', reason='Not all authorized cohort rows completed.', saved_prediction_rows=len(pred))
    if not completed:
        return None, dict(status='incomplete', reason='Fixed 200-epoch endpoint/completion evidence missing.', saved_prediction_rows=len(pred))
    return pred, dict(status='independently_rescored', n_original_images=len(pred),
                      global_weighted_r2_raw=raw_weighted_r2(pred[TARGETS], pred[PREDICTED]),
                      negative_prediction_rate=0.0, prediction_sha256=_sha(path))


def _verify_protocol(root, frame, predictions):
    """Check reported paired exposure without loading or modifying model weights."""
    audit = {'status': 'verified' if predictions else 'no_complete_arms', 'folds': {}}
    fields = ['initial_model_sha256', 'raw_scale_training_only_g', 'median_training_only_g',
              'train_image_ids', 'validation_image_ids', 'validation_cache']
    source_hashes = set()
    for fold in FOLDS:
        configs = {arm: _read_json(root/'arms'/arm/f'fold_{fold}'/'config.json') for arm in predictions}
        if not configs or not all(configs.values()):
            audit['status'] = 'incomplete_provenance'
            audit['folds'][str(fold)] = {'status': 'missing_config', 'arms': [a for a,c in configs.items() if not c]}
            continue
        baseline = configs.get('control', next(iter(configs.values())))
        steps = {}
        curves = {}
        for arm, config in configs.items():
            if any(field not in config for field in fields):
                raise ValueError('Completed arm config lacks paired initialization/split/scale evidence.')
            if any(config[field] != baseline[field] for field in fields):
                raise ValueError('Paired head initialization, scales or original image splits differ.')
            if set(config['train_image_ids']) != set(frame[frame.fold != fold].image_path):
                raise ValueError('Training image identities differ from original fold split.')
            if set(config['validation_image_ids']) != set(frame[frame.fold == fold].image_path):
                raise ValueError('Validation image identities differ from original fold split.')
            if config['validation_cache'] != 'control_orientation_0':
                raise ValueError('Validation must use unshuffled identity-orientation control features.')
            signature = config.get('signature', {})
            expected = {'folds': list(FOLDS), 'epochs': 200, 'swa_last_epochs': 40, 'seed': 42,
                'batch_size': 32, 'lr': 3e-4, 'weight_decay': .01, 'width': 512, 'dropout': .2,
                'objective': 'raw_global_weighted_mse_one_training_scale',
                'selection': 'fixed_final_swa_endpoint_not_best_validation_epoch'}
            if any(signature.get(key) != value for key,value in expected.items()):
                raise ValueError('Completed head differs from the fixed matched training recipe.')
            if arm != 'control' and (signature.get('shuffle_grid') != (4 if arm=='shuffle_4' else 10)
                                    or signature.get('shuffle_probability') != .35):
                raise ValueError('Treatment differs from its predeclared shuffle probability/grid.')
            if arm == 'control' and signature.get('shuffle_grid', 0) != 0:
                raise ValueError('Control is shuffled.')
            origin = config.get('feature_origin', {})
            if origin.get('representation') != 'encoder' or origin.get('grass_steps') != 2400 or not origin.get('source_sha256'):
                raise ValueError('Need reported completed grass-adapted student encoder provenance.')
            source_hashes.add(origin['source_sha256'])
            if origin != baseline.get('feature_origin'):
                raise ValueError('Paired arms use different frozen source or preprocessing.')
            baseline_signature = {k:v for k,v in baseline['signature'].items() if not k.startswith('shuffle_')}
            observed_signature = {k:v for k,v in signature.items() if not k.startswith('shuffle_')}
            if observed_signature != baseline_signature:
                raise ValueError('Paired arms differ in base training signature.')
            metrics = _read_json(root/'arms'/arm/f'fold_{fold}'/'metrics.json')
            count = metrics.get('optimizer_steps')
            expected_count = 200 * ((len(config['train_image_ids']) + 31) // 32)
            if count != expected_count or metrics.get('initial_model_sha256') != config['initial_model_sha256']:
                raise ValueError('Optimizer counts or initial model identity differ from fixed endpoint.')
            steps[arm] = count
            path=root/'arms'/arm/f'fold_{fold}'/'train_log.jsonl'
            rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []
            complete_log = (len(rows) == 200 and [row.get('epoch') for row in rows] == list(range(1,201))
                and all('lr' in row and 'optimizer_steps' in row for row in rows))
            if complete_log:
                curves[arm] = [(row['epoch'],row['lr'],row['optimizer_steps']) for row in rows]
                per_epoch_steps = (len(config['train_image_ids']) + 31) // 32
                if any(not np.isfinite(row['lr']) or row['lr'] < 0 or row['optimizer_steps'] != row['epoch']*per_epoch_steps for row in rows):
                    raise ValueError('Per-epoch learning-rate/update-count evidence is invalid.')
            runtime=config.get('runtime',{})
            if runtime and (runtime.get('encoder_frozen') is not True or runtime.get('predictor_frozen') is not True):
                raise ValueError('Frozen source runtime is not reported frozen.')
            tiny = signature.get('tiny') is True and origin.get('tiny') is True
            if not tiny:
                gpu = ''.join(c for c in str(runtime.get('gpu', '')).lower() if c.isalnum())
                precision = runtime.get('precision', {})
                if not isinstance(precision, dict):
                    precision = {}
                layers = precision.get('float8_layers', 0)
                if (signature.get('precision') != 'fp8' or origin.get('precision') != 'fp8'
                        or precision.get('requested') != 'fp8'
                        or precision.get('active') != 'fp8_linear_bf16_other'
                        or type(layers) is not int or layers <= 0
                        or 'rtxpro6000' not in gpu or 'blackwell' not in gpu):
                    raise ValueError('Real protocol requires actual RTX PRO6000 Blackwell and active FP8 runtime/cache provenance.')
        if len(source_hashes) != 1:
            raise ValueError('Different adapted sources used across arms/folds.')
        if curves and len(curves) == len(configs) and any(curve != next(iter(curves.values())) for curve in curves.values()):
            raise ValueError('Paired learning rates or optimizer counts differ in logs.')
        complete_logs = len(curves) == len(configs)
        if not complete_logs:
            audit['status'] = 'incomplete_provenance'
        audit['folds'][str(fold)] = {'status':'verified_reported_protocol' if complete_logs else 'incomplete_epoch_log_provenance',
            'optimizer_steps':steps, 'matched_lr_log_evidence':complete_logs,
            'tiny_software_smoke':all(c['signature'].get('tiny') is True and c['feature_origin'].get('tiny') is True for c in configs.values()),
            'incomplete_epoch_log_arms':[arm for arm in configs if arm not in curves]}
    audit['reported_source_sha256'] = sorted(source_hashes)
    audit['scope'] = 'Checks reported configs/logs and independently matched rows; does not load checkpoint tensors or independently verify remote hardware.'
    return audit


def _tables(frame, predictions, reports):
    targets, folds, errors, distributions, split_counts = [], [], [], [], []
    eval_frame = frame[frame.fold.isin(FOLDS)]
    for target in TARGETS:
        values = eval_frame[target].to_numpy(float)
        distributions.append(dict(target=target, n=len(values), mean_g=float(values.mean()),
            min_g=float(values.min()), max_g=float(values.max()), median_g=float(np.median(values))))
    for fold in FOLDS:
        train, evaluation = frame[frame.fold != fold], frame[frame.fold == fold]
        split_counts.append(dict(fold=fold, train_original_images=len(train), eval_original_images=len(evaluation),
            train_date_groups=train.group_id.nunique(), eval_date_groups=evaluation.group_id.nunique(),
            overlapping_groups=len(set(train.group_id) & set(evaluation.group_id))))
    for arm, pred in predictions.items():
        y, p = pred[TARGETS].to_numpy(float), pred[PREDICTED].to_numpy(float)
        for j, target in enumerate(TARGETS):
            denominator = np.square(y[:, j] - y[:, j].mean()).sum()
            targets.append(dict(arm=arm, target=target, n=len(y), mae_g=float(np.abs(y[:, j] - p[:, j]).mean()),
                rmse_g=float(np.sqrt(np.square(y[:, j] - p[:, j]).mean())),
                r2_raw=float(1 - np.square(y[:, j] - p[:, j]).sum()/denominator) if denominator > 0 else None))
        for fold, subset in pred.groupby('fold'):
            fy, fp = subset[TARGETS].to_numpy(float), subset[PREDICTED].to_numpy(float)
            folds.append(dict(arm=arm, fold=int(fold), n=len(subset), date_groups=subset.group_id.nunique(),
                global_weighted_r2_raw=raw_weighted_r2(fy, fp), weighted_mae_g=float((np.abs(fy-fp)*WEIGHTS).sum()/len(fy)),
                weighted_rmse_g=float(np.sqrt((np.square(fy-fp)*WEIGHTS).sum()/len(fy)))))
        cutoff = float(np.quantile(y[:, 4], .9))
        for i in range(len(pred)):
            errors.append(dict(arm=arm, sample_index=i, true_total_g=float(y[i, 4]),
                high_total_decile=bool(y[i, 4] >= cutoff), **{f'absolute_error_{t}':float(abs(y[i,j]-p[i,j])) for j,t in enumerate(TARGETS)}))
    for name, values in [('metrics_by_target', targets), ('metrics_by_fold', folds), ('sample_errors', errors),
                         ('target_distributions', distributions), ('train_eval_counts', split_counts)]:
        pd.DataFrame(values).to_csv(reports / (name + '.csv'), index=False)
    frame.groupby('group_id').size().rename('n_original_images').to_csv(reports/'group_sizes.csv')
    residual = eval_frame.Dry_Total_g - eval_frame.GDM_g - eval_frame.Dry_Dead_g
    other = eval_frame.GDM_g - eval_frame.Dry_Green_g - eval_frame.Dry_Clover_g
    return dict(total_sum_discrepancies_gt_015g=int((residual.abs() > .15).sum()),
                max_abs_total_sum_residual_g=float(residual.abs().max()),
                max_abs_gdm_sum_residual_g=float(other.abs().max()), original_labels_preserved=True)


def _logs(path, xkey, ykey):
    values = []
    if Path(path).is_file():
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if xkey in row and ykey in row:
                x, y = float(row[xkey]), float(row[ykey])
                if np.isfinite(x) and np.isfinite(y):
                    values.append((x, y))
    return values


def _figures(root, frame, predictions, results, examples, ssl_log):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps

    figures = root/'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'svg.fonttype': 'none'})
    statuses = {}

    def save(fig, name):
        fig.savefig(figures/(name+'.svg'), bbox_inches='tight')
        fig.savefig(figures/(name+'.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        statuses[name] = dict(status='rendered')

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    boxes = [(0.07, .65, 'One original field\n2000 × 1000 pixels'), (.28, .65, 'Two nonoverlapping\n1000 × 1000 halves'),
             (.51, .65, 'Training only: control /\n4×4 / 10×10 permutation\np = 0.35; then 224² + quarter-turn'),
             (.77, .65, 'Frozen grass-adapted\nI-JEPA ViT-H/14 encoder'), (.77, .23, 'Two mean view features\nSame gated regression head'),
             (.51, .23, 'Green / Dead / Clover\nGDM = Green + Clover\nTotal = GDM + Dead'),
             (.28, .23, 'Validation / inference\nOriginal unshuffled halves\nIdentity orientation')]
    for x, y, label in boxes:
        ax.text(x, y, label, ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=.6', facecolor='#edf3f6', edgecolor='#526976'))
    for start, end in [(0,1),(1,2),(2,3),(3,4),(4,5),(0,6),(6,3)]:
        x1,y1,_=boxes[start]; x2,y2,_=boxes[end]
        ax.annotate('', xy=(x2-.07 if x2>x1 else x2+.07,y2), xytext=(x1+.07 if x2>x1 else x1-.07,y1),
                    xycoords='axes fraction', arrowprops=dict(arrowstyle='->', color='#526976'))
    fig.suptitle('Figure 1 — Fixed matched protocol; rearrangements do not add biological measurements')
    _json(figures/'figure_01_architecture_source.json', {'nodes':[b[2] for b in boxes], 'training_shuffle_probability':.35, 'validation_shuffle':False})
    save(fig, 'figure_01_architecture')

    statuses['figure_02_examples'] = dict(status='disabled' if not examples else 'missing', reason='No verified full source photographs available.')
    example_sources = []
    if examples and 'absolute_path' in frame:
        from .block_shuffle import permutation_seed, shuffle_square_tile
        candidates = frame[frame.absolute_path.map(lambda p: isinstance(p,str) and Path(p).is_file())].iloc[:3]
        if len(candidates) >= 2:
            fig, axes = plt.subplots(len(candidates), 3, figsize=(12, 3*len(candidates)), squeeze=False)
            for row_index, (_, row) in enumerate(candidates.iterrows()):
                path = Path(row.absolute_path)
                if _sha(path) != row.image_sha256:
                    raise ValueError('Example photo digest differs from prepared source.')
                with Image.open(path) as original:
                    image = ImageOps.exif_transpose(original).convert('RGB')
                if image.size != (2000, 1000):
                    raise ValueError('Example requires the original 2000 × 1000 field photograph.')
                source = dict(sample_index=int(row.name), image_path=row.image_path, image_sha256=row.image_sha256, variant=0, synthetic_columns=[4,10], seeds={})
                for col, grid in enumerate([0,4,10]):
                    output = image.copy()
                    if grid:
                        source['seeds'][str(grid)] = []
                        for view in range(2):
                            tile=image.crop((1000*view,0,1000*(view+1),1000))
                            permseed=permutation_seed(row.image_sha256,grid,0,view)
                            source['seeds'][str(grid)].append(permseed)
                            output.paste(shuffle_square_tile(tile,grid,permseed),(1000*view,0))
                    axes[row_index,col].imshow(output)
                    axes[row_index,col].axis('off')
                    axes[row_index,col].set_title(f'Source index {row.name}: ' + ('original' if not grid else f'{grid}×{grid} synthetic rearrangement'))
                example_sources.append(source)
            fig.suptitle('Figure 2 — Real source photographs and synthetic within-half rearrangements')
            save(fig, 'figure_02_examples')
    _json(figures/'figure_02_examples_source.json', {'status':statuses['figure_02_examples'], 'sources':example_sources, 'caption':'Synthetic rearrangements; no additional labeled field measurements.'})

    fig, axes = plt.subplots(1,2,figsize=(10,4))
    arms = list(predictions)
    if arms:
        axes[0].bar(arms, [results['arms'][a]['global_weighted_r2_raw'] if results['arms'][a]['global_weighted_r2_raw'] is not None else np.nan for a in arms], color='#587b90')
    else:
        axes[0].text(.5,.5,'No completed matched arms',ha='center',transform=axes[0].transAxes)
    axes[0].set_ylabel('Pooled global raw weighted R²')
    axes[0].tick_params(axis='x',rotation=20)
    for i,(name,row) in enumerate(results['comparisons'].items()):
        if row['ci95'] is not None:
            lo,hi=row['ci95']; effect=row['observed_difference']
            axes[1].plot([lo,hi],[i,i],color='#587b90',lw=2)
            axes[1].plot(effect,i,'o',color='#263e4c')
    axes[1].set_yticks(range(len(results['comparisons'])), list(results['comparisons']))
    axes[1].axvline(0,color='black',ls='--',lw=.8)
    axes[1].set_xlabel('Paired Δ pooled raw weighted R²; percentile 95% CI')
    fig.suptitle(f"Figure 3 — {results['cohort']['scope']}: N={results['cohort']['n_original_images']}, "
                 f"{results['cohort']['independent_date_groups']} date groups, one training seed")
    fig.tight_layout()
    _json(figures/'figure_03_paired_effects_source.json', {'arms':results['arms'], 'comparisons':results['comparisons'], 'cohort':results['cohort']})
    save(fig,'figure_03_paired_effects')

    if predictions:
        fig, axes = plt.subplots(5,len(predictions),figsize=(4*len(predictions),15),squeeze=False)
        for j,target in enumerate(TARGETS):
            max_value=max(max(float(pred[target].max()),float(pred['pred_'+target].max())) for pred in predictions.values())*1.05
            for col,(arm,pred) in enumerate(predictions.items()):
                ax=axes[j,col]
                ax.scatter(pred[target],pred['pred_'+target],s=10,alpha=.65)
                ax.plot([0,max_value],[0,max_value],'k--',lw=.8)
                ax.set(xlim=(0,max_value),ylim=(0,max_value),xlabel='Measured (g)',ylabel='Predicted (g)',title=f'{arm}: {target}')
        fig.suptitle('Figure 4 — Matched original images; common axes within each target')
        fig.tight_layout()
        save(fig,'figure_04_per_target_scatter')
        fig, axes = plt.subplots(2,2,figsize=(10,8))
        for ax,j in zip(axes.flat,[4,0,1,2]):
            for arm,pred in predictions.items():
                err=np.abs(pred[TARGETS[j]].to_numpy()-pred[PREDICTED[j]].to_numpy())
                ax.scatter(pred.Dry_Total_g,err,s=12,alpha=.65,label=arm)
                index=int(np.argmax(err))
                ax.annotate(str(index),(pred.Dry_Total_g.iloc[index],err[index]),fontsize=7)
            ax.set(xlabel='Measured total biomass (g)',ylabel=f'Absolute {TARGETS[j]} error (g)')
        axes.flat[0].legend()
        fig.suptitle('Figure 5 — Raw residuals; largest errors labeled only by sample index; no smoothing')
        fig.tight_layout()
        save(fig,'figure_05_residuals')
        for name in ['figure_04_per_target_scatter','figure_05_residuals']:
            pd.concat([pred.assign(arm=arm, sample_index=np.arange(len(pred))) for arm,pred in predictions.items()]).to_csv(figures/(name+'_source.csv'),index=False)
    else:
        for name in ['figure_04_per_target_scatter','figure_05_residuals']:
            statuses[name]=dict(status='missing',reason='No completed arm predictions.')

    fig, axes = plt.subplots(1,2,figsize=(11,4))
    head_rows=[]
    for arm in ARMS:
        for path in sorted((root/'arms'/arm).glob('fold_*/train_log.jsonl')):
            values=_logs(path,'epoch','loss')
            if values:
                axes[0].plot(*np.asarray(values).T,label=f'{arm}/{path.parent.name}')
                head_rows.extend(dict(arm=arm,fold=path.parent.name,epoch=x,loss=y,source_sha256=_sha(path)) for x,y in values)
    if head_rows:
        axes[0].legend(fontsize=6)
    else:
        axes[0].text(.5,.5,'Head training logs missing',ha='center',transform=axes[0].transAxes)
    axes[0].set(xlabel='Head epoch (logged endpoints)',ylabel='Training raw weighted head loss',title='Supervised head training')
    ssl_values = _logs(ssl_log,'step','loss') if ssl_log else []
    ssl_info=dict(status='available' if ssl_values else 'missing', source_sha256=_sha(ssl_log) if ssl_log and Path(ssl_log).is_file() else None,
                  interpretation='Historical SSL, not new pretraining. Latent loss does not establish downstream R² improvement.')
    if ssl_values:
        axes[1].plot(*np.asarray(ssl_values).T,color='#587b90')
    else:
        axes[1].text(.5,.5,'Historical SSL log missing',ha='center',transform=axes[1].transAxes)
    axes[1].set(xlabel='Original historical SSL step',ylabel='Historical latent loss',title='Historical JEPA SSL; separate objective')
    fig.suptitle('Figure 6 — Actual logged training dynamics')
    fig.tight_layout()
    pd.DataFrame(head_rows).to_csv(figures/'figure_06_head_loss_source.csv',index=False)
    pd.DataFrame(ssl_values,columns=['step','loss']).to_csv(figures/'figure_06_historical_ssl_source.csv',index=False)
    save(fig,'figure_06_learning_curves')
    fig, ax=plt.subplots(figsize=(6,4))
    ax.hist(frame.groupby('group_id').size().to_numpy(),bins='auto',edgecolor='white',color='#587b90')
    ax.set(xlabel='Original images per acquisition-date group',ylabel='Group count',title='Prepared cohort group sizes; dates are site proxies')
    save(fig,'supplementary_group_sizes')
    return statuses, dict(head=dict(status='available' if head_rows else 'missing', logged_points=len(head_rows)), historical_ssl=ssl_info)


def report(root, folds_csv, draws=10000, seed=42, ssl_log=None, examples=True):
    """Rescore, verify pairing, write actual-data figures and a weight-free archive."""
    root, folds_csv = Path(root).resolve(), Path(folds_csv).resolve()
    frame = _validate_prepared(folds_csv)
    eval_frame = frame[frame.fold.isin(FOLDS)]
    if not len(eval_frame) or set(eval_frame.fold) != set(FOLDS):
        raise ValueError('Prepared cohort must contain all authorized folds 0/1/2.')
    predictions, arm_info = {}, {}
    for arm in ARMS:
        pred, info = _load_arm(root/'arms'/arm,frame)
        arm_info[arm] = info
        if pred is not None:
            predictions[arm] = pred
    protocol_audit = _verify_protocol(root, frame, predictions)
    y = eval_frame[TARGETS].to_numpy(float)
    samples, comparisons = paired_bootstrap(y,{a:p[PREDICTED].to_numpy(float) for a,p in predictions.items()},
                                          eval_frame.group_id.to_numpy(),draws=draws,seed=seed)
    reports = root/'reports'
    reports.mkdir(parents=True,exist_ok=True)
    samples.to_csv(reports/'paired_bootstrap.csv',index=False)
    result = dict(arms=arm_info,comparisons=comparisons,limits=LIMITS,protocol_audit=protocol_audit,
        cohort=dict(scope='partial development only', authorized_folds=list(FOLDS), n_original_images=len(eval_frame),
                    prepared_original_images=len(frame), independent_date_groups=eval_frame.group_id.nunique(),
                    n_labeled_biological_measurements=len(frame), full_oof=False, training_seeds=1),
        bootstrap=dict(draws=draws,seed=seed),
        label_sum_audit=_tables(frame,predictions,reports))
    tiny_smoke = bool(protocol_audit['folds']) and all(row.get('tiny_software_smoke') is True for row in protocol_audit['folds'].values())
    result['scientific_results'] = not tiny_smoke
    if tiny_smoke:
        result['cohort']['scope'] = 'tiny synthetic software smoke only'
        result['limits'] = 'Synthetic tiny CPU software smoke; numerical outputs are software checks, not scientific results. ' + LIMITS
        for comparison in result['comparisons'].values():
            comparison['interpretation'] = 'Synthetic software smoke; no scientific inference.'
            comparison['ci_supports_positive'] = False
    provenance = _read_json(root/'PROVENANCE.json')
    try:
        revision=subprocess.run(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,check=True).stdout.strip()
    except (subprocess.SubprocessError,OSError):
        revision=None
    inventory=[]
    for directory in [root/'arms', root/'features']:
        for path in sorted(directory.rglob('*')) if directory.is_dir() else []:
            if not path.is_file():
                continue
            if path.suffix in {'.json','.jsonl','.csv'} or path.name=='selected.pt':
                inventory.append(dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=_sha(path)))
    result['provenance'] = dict(analysis_datetime_utc=datetime.now(timezone.utc).isoformat(),
        code_git_sha=revision,report_module_sha256=_sha(__file__),folds_csv_sha256=_sha(folds_csv),
        upstream_provenance_status='present' if provenance else 'missing', upstream=provenance,
        checkpoint_hash_scope='Selected head bytes only; frozen source/cache bytes not loaded or reverified by this analysis.',
        artifact_inventory=inventory)
    _json(reports/'analysis_provenance.json',result['provenance'])
    result['figures'], result['learning_dynamics'] = _figures(root,frame,predictions,result,examples,ssl_log)
    result['publication_ready'] = False
    result['publication_readiness_note'] = 'Requires coordinator review, verified source/runtime provenance, preserved outputs, licenses and data/code availability statements; figures alone do not establish readiness.'
    _json(reports/'paired_comparisons.json',result)
    _json(reports/'figure_status.json',result['figures'])
    (reports/'interpretation.md').write_text('\n'.join(['# Spatial permutation ablation',result['limits'],'',
        *[f"- {arm}: {info['status']}" for arm,info in arm_info.items()], '',
        *[f"- {key}: {row['interpretation']}" for key,row in comparisons.items()], '',
        'Data availability: CSIRO competition data remain subject to the original dataset terms; no private test labels were used.',
        'Code availability: the controlled experiment and independent analysis live in the private Img2BioMass repository; public release status is not asserted.',
        'Prior art: Meta I-JEPA official source; Noroozi & Favaro, ECCV 2016 (jigsaw learning). This is not a claim of novel jigsaw learning.',
        'Meta reference code/checkpoint licensing, including CC BY-NC 4.0, must accompany any authorized distribution.'])+'\n')
    source_dir=reports/'code'
    source_dir.mkdir(exist_ok=True)
    for name in ['block_shuffle_report.py','block_shuffle.py','official_heads.py','data.py','metrics.py']:
        source=Path(__file__).parent/name
        if source.is_file():
            shutil.copyfile(source,source_dir/name)
    archive=root/'paper_results.zip'
    allowed={'.py','.json','.jsonl','.csv','.md','.svg','.png'}
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in allowed and path.stat().st_size<=20*1024*1024:
                relative=path.relative_to(root)
                if 'features' not in relative.parts and not path.is_symlink():
                    handle.write(path,str(relative))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--folds-csv',required=True)
    parser.add_argument('--draws',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--ssl-log')
    parser.add_argument('--examples',action='store_true',help='Render only verified original source photos if available.')
    args=parser.parse_args()
    result=report(args.root,args.folds_csv,args.draws,args.seed,args.ssl_log,args.examples)
    print(json.dumps({'arms':result['arms'],'comparisons':result['comparisons'],'scope':result['cohort']['scope']},allow_nan=False))


if __name__=='__main__':
    main()
