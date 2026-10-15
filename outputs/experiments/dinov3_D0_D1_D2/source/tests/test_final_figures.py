"""Synthetic report fixtures are functional tests, never research evidence."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from biomass_jepa.data import TARGETS, file_sha256
from biomass_jepa.final_figures import ARMS, FIGURES, report
from biomass_jepa.final_metrics import global_r2


def fixture(tmp_path):
    image = tmp_path / 'synthetic_only.png'
    pixels = np.zeros((1000, 2000, 3), dtype=np.uint8)
    pixels[:, :1000, 0] = np.arange(1000, dtype=np.uint16)[:, None] % 256
    pixels[:, 1000:, 1] = np.arange(1000, dtype=np.uint16)[None, :] % 256
    Image.fromarray(pixels).save(image)
    records = []
    for i in range(357):
        if i < 49: fold, group = 0, i % 4
        elif i < 105: fold, group = 1, 4 + (i - 49) % 5
        else: fold, group = 2, 9 + (i - 105) % 19
        green, dead, clover = 10 + i % 30, 2 + i % 7, 3 + i % 13
        y = [green, dead, clover, green + clover, green + dead + clover]
        records.append({'image_path': f'{i:04d}.png', 'absolute_path': str(image),
                        'image_sha256': file_sha256(image), 'fold': fold,
                        'Sampling_Date': f'date{group:02d}', **dict(zip(TARGETS, y))})
    frame = pd.DataFrame(records); root = tmp_path / 'synthetic_report'
    for branch, names in ARMS.items():
        for index, arm in enumerate(names):
            for fold in (0, 1):
                directory = root / branch / f'fold_{fold}' / arm; directory.mkdir(parents=True)
                rows = frame[frame.fold == fold][['image_path', 'Sampling_Date', 'fold'] + TARGETS].copy()
                pred = rows[TARGETS].to_numpy(np.float64) + .1 * (index + 1)
                for j, target in enumerate(TARGETS):
                    rows['pred_' + target] = pred[:, j]
                    if arm == 'D2': rows['coarse_' + target] = pred[:, j] + .03
                rows.to_csv(directory / 'val_predictions.csv', index=False, float_format='%.17g')
                serialized = pd.read_csv(directory / 'val_predictions.csv')
                checkpoint = directory / 'head_selected.pt'; checkpoint.write_bytes(b'synthetic-not-a-model')
                metrics = {'complete': True, 'completed_epochs': 200, 'swa_count': 40,
                           'head_sha256': file_sha256(checkpoint),
                           'global_weighted_r2_raw': global_r2(serialized[TARGETS], serialized[['pred_' + t for t in TARGETS]]),
                           'parameters': 100, 'elapsed_seconds': 5., 'optimizer_steps': 2000,
                           'precision': {'active': 'synthetic test only'}}
                (directory / 'metrics.json').write_text(json.dumps(metrics))
                pd.DataFrame({'epoch': np.arange(1, 201), 'loss': np.ones(200),
                              'steps': np.arange(1, 201) * 10}).to_csv(directory / 'head_training.csv', index=False)
                if branch == 'jepa':
                    updates = 385 if fold == 0 else 377
                    pd.DataFrame({'step': np.arange(1, updates + 1), 'loss': np.ones(updates),
                                  'elapsed_seconds': np.arange(1, updates + 1)}).to_csv(directory / 'ssl_training.csv', index=False)
    return root, frame


def test_complete_report_alignment_source_data_and_figures(tmp_path):
    root, frame = fixture(tmp_path)
    outcome = report(root, frame)
    assert outcome['status'] == 'completed_reporting_pending_independent_P1'
    assert len(outcome['main_results']) == 5 and len(outcome['paired']) == 3
    assert all(row['effective_draws'] == 10000 for row in outcome['paired'])
    assert all(row['metric_parity_error'] <= 1e-10 for row in outcome['main_results'])
    for name in FIGURES: assert (root / 'figures' / (name + '.png')).stat().st_size > 1000
    for branch in ARMS:
        oof = pd.read_csv(root / branch / 'paired_oof.csv')
        assert len(oof) == 105 and oof.image_path.nunique() == 105
    manifest = pd.read_csv(root / 'figure_data/jigsaw_panel_manifest.csv')
    assert len(manifest) == 2
    assert all(sorted(json.loads(order)) == list(range(100)) for order in manifest.permutation)
    assert len(pd.read_csv(root / 'stats/per_target.csv')) == 25
    assert len(pd.read_csv(root / 'figure_data/residuals.csv')) == 5 * 5 * 105
    text = (root / 'FINAL_REPORT.md').read_text()
    assert 'neither full OOF nor competition scores' in text
    assert 'P1' in text and 'no heatmaps were invented' in text


@pytest.mark.parametrize('corruption', ['missing', 'incomplete', 'identity', 'labels', 'negative', 'metric', 'checkpoint', 'logs', 'coarse'])
def test_report_rejects_incomplete_or_misaligned_evidence_before_figures(tmp_path, corruption):
    root, frame = fixture(tmp_path)
    directory = root / 'dinov3/fold_0/D2'
    if corruption == 'missing': (directory / 'val_predictions.csv').unlink()
    elif corruption in ('incomplete', 'metric'):
        path = directory / 'metrics.json'; metrics = json.loads(path.read_text())
        metrics['complete' if corruption == 'incomplete' else 'global_weighted_r2_raw'] = False if corruption == 'incomplete' else -9
        path.write_text(json.dumps(metrics))
    elif corruption == 'checkpoint': (directory / 'head_selected.pt').write_bytes(b'altered')
    elif corruption == 'logs':
        path = directory / 'head_training.csv'; pd.read_csv(path).iloc[:-1].to_csv(path, index=False)
    else:
        path = directory / 'val_predictions.csv'; rows = pd.read_csv(path)
        if corruption == 'identity': rows.loc[0, 'Sampling_Date'] = 'wrong'
        if corruption == 'labels': rows.loc[0, TARGETS[0]] += 1
        if corruption == 'negative': rows.loc[0, 'pred_' + TARGETS[0]] = -1
        if corruption == 'coarse': rows.drop(columns=['coarse_' + TARGETS[0]], inplace=True)
        rows.to_csv(path, index=False)
    with pytest.raises((ValueError, FileNotFoundError)): report(root, frame)
    assert not (root / 'FINAL_REPORT.md').exists()
    assert not (root / 'figures').exists()
