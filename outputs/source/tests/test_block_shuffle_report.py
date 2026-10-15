"""Independent rescore guards: changing pairing or accepting bad rows fails these tests."""
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TARGETS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']


def fixture_run(tmp_path, omit_c=False):
    root = tmp_path / 'final_one_run'
    rows = []
    for i in range(12):
        green, dead, clover = 1.0 + i * 2, 2.0 + i, 1.0 + i / 2
        rows.append(dict(image_path=f'image_{i}.jpg', group_id=f'date_{i // 2}',
                         Sampling_Date=f'date_{i // 2}', fold=(i // 2) % 3,
                         image_sha256=f'{i:064x}',
                         **dict(zip(TARGETS, [green, dead, clover, green + clover, green + clover + dead]))))
    frame = pd.DataFrame(rows)
    path = tmp_path / 'prepared.csv'
    frame.to_csv(path, index=False)
    for arm, error in [('control', 3), ('shuffle_4', 1), ('shuffle_10', 5)]:
        folder = root / 'arms' / arm
        folder.mkdir(parents=True)
        if arm == 'shuffle_10' and omit_c:
            continue
        pred = frame[['image_path', 'group_id', 'fold'] + TARGETS].copy()
        for name in TARGETS[:3]:
            pred[f'pred_{name}'] = pred[name] + error
        pred['pred_GDM_g'] = pred.pred_Dry_Green_g + pred.pred_Dry_Clover_g
        pred['pred_Dry_Total_g'] = pred.pred_GDM_g + pred.pred_Dry_Dead_g
        pred.to_csv(folder / 'oof_predictions.csv', index=False)
        (folder / 'metrics.json').write_text(json.dumps(dict(status='completed_heads', folds=[0, 1, 2])))
        for fold in range(3):
            fdir = folder / f'fold_{fold}'
            fdir.mkdir()
            (fdir / 'metrics.json').write_text(json.dumps(dict(status='completed_fixed_endpoint', completed_epochs=200, primary_selection='final_swa')))
            pred[pred.fold == fold].to_csv(fdir / 'predictions.csv', index=False)
            (fdir / 'train_log.jsonl').write_text('\n'.join(json.dumps(dict(epoch=e, loss=4/e)) for e in [10, 20, 30]))
            (fdir / 'selected.pt').write_bytes(b'not-real-checkpoint-fixture')
    return root, path, frame


def test_independent_pooled_metric_and_paired_group_resampling():
    from biomass_jepa.block_shuffle_report import raw_weighted_r2, paired_bootstrap
    y = np.array([[1, 2, 3, 4, 6], [10, 20, 30, 40, 60], [2, 3, 4, 6, 9]], float)
    p = y + 2
    weights = np.array([.1, .1, .1, .2, .5])
    mean = (y * weights).sum() / len(y)
    expected = 1 - ((y - p)**2 * weights).sum() / (((y - mean)**2) * weights).sum()
    assert raw_weighted_r2(y, p) == pytest.approx(expected)
    samples, comparisons = paired_bootstrap(y, {'control': p, 'shuffle_4': p.copy()}, np.array(['a', 'a', 'b']), draws=40, seed=42)
    assert len(samples) == 40
    assert np.array_equal(samples.control, samples.shuffle_4)
    assert comparisons['shuffle_4_minus_control']['ci95'] == [0.0, 0.0]
    assert comparisons['shuffle_4_minus_control']['independent_date_groups'] == 2
    again, _ = paired_bootstrap(y, {'control': p, 'shuffle_4': p.copy()}, np.array(['a', 'a', 'b']), draws=40, seed=42)
    pd.testing.assert_frame_equal(samples, again)


def test_report_outputs_and_incomplete_arm_are_honest(tmp_path):
    from biomass_jepa.block_shuffle_report import report
    root, folds, _ = fixture_run(tmp_path, omit_c=True)
    result = report(root, folds, draws=50, seed=42, examples=False)
    assert result['arms']['shuffle_10']['status'] == 'incomplete'
    assert set(result['comparisons']) == {'shuffle_4_minus_control'}
    assert result['cohort']['scope'] == 'partial development only'
    assert result['cohort']['n_original_images'] == 12
    assert result['comparisons']['shuffle_4_minus_control']['observed_difference'] > 0
    assert (root/'reports/paired_bootstrap.csv').is_file()
    for number, name in [(1, 'architecture'), (3, 'paired_effects'), (4, 'per_target_scatter'), (5, 'residuals'), (6, 'learning_curves')]:
        for ext in ['svg', 'png']:
            assert (root/f'figures/figure_{number:02d}_{name}.{ext}').is_file()
    assert result['figures']['figure_02_examples']['status'] == 'disabled'
    assert result['learning_dynamics']['historical_ssl']['status'] == 'missing'
    with zipfile.ZipFile(root/'paper_results.zip') as archive:
        names = archive.namelist()
        assert 'reports/paired_comparisons.json' in names
        assert any(n.endswith('train_log.jsonl') for n in names)
        assert not any(n.endswith('.pt') for n in names)


@pytest.mark.parametrize('bad', ['duplicate', 'labels', 'groups', 'negative', 'nan', 'missing_row'])
def test_report_rejects_unmatched_or_invalid_predictions(tmp_path, bad):
    from biomass_jepa.block_shuffle_report import report
    root, folds, _ = fixture_run(tmp_path)
    path = root/'arms/shuffle_4/oof_predictions.csv'
    pred = pd.read_csv(path)
    if bad == 'duplicate':
        pred = pd.concat([pred, pred.iloc[:1]])
    elif bad == 'labels':
        pred.loc[0, TARGETS[0]] += 1
    elif bad == 'groups':
        pred.loc[0, 'group_id'] = 'foreign-date'
    elif bad == 'negative':
        pred.loc[0, 'pred_' + TARGETS[0]] = -1
    elif bad == 'nan':
        pred.loc[0, 'pred_' + TARGETS[0]] = np.nan
    elif bad == 'missing_row':
        pred = pred.iloc[1:]
    pred.to_csv(path, index=False)
    with pytest.raises(ValueError):
        report(root, folds, draws=5, examples=False)


def test_report_rejects_date_groups_crossing_folds(tmp_path):
    from biomass_jepa.block_shuffle_report import report
    root, folds, frame = fixture_run(tmp_path)
    frame.loc[0, 'fold'] = 1
    frame.to_csv(folds, index=False)
    with pytest.raises(ValueError, match='cross'):
        report(root, folds, draws=5, examples=False)


def test_degenerate_metric_has_no_invented_score():
    from biomass_jepa.block_shuffle_report import raw_weighted_r2
    assert raw_weighted_r2(np.ones((2, 5)), np.ones((2, 5))) is None


def add_protocol_configs(root, frame):
    signature = dict(folds=[0,1,2], epochs=200, swa_last_epochs=40, seed=42, batch_size=32,
                     lr=3e-4, weight_decay=.01, width=512, dropout=.2, precision='fp8',
                     selection='fixed_final_swa_endpoint_not_best_validation_epoch',
                     objective='raw_global_weighted_mse_one_training_scale')
    for arm in ['control', 'shuffle_4', 'shuffle_10']:
        for fold in range(3):
            folder = root/'arms'/arm/f'fold_{fold}'
            config = dict(signature=signature.copy(), feature_origin=dict(source_sha256='a'*64, representation='encoder', grass_steps=2400, precision='fp8'),
                initial_model_sha256=f'{fold:064x}', raw_scale_training_only_g=50.+fold,
                median_training_only_g=[1.,2.,3.], validation_cache='control_orientation_0',
                train_image_ids=frame[frame.fold!=fold].image_path.tolist(),
                validation_image_ids=frame[frame.fold==fold].image_path.tolist(),
                runtime=dict(gpu='NVIDIA RTX PRO 6000 Blackwell', precision=dict(requested='fp8',active='fp8_linear_bf16_other',float8_layers=4,teacher='bf16'), encoder_frozen=True, predictor_frozen=True))
            if arm != 'control':
                config['signature'].update(shuffle_grid=4 if arm=='shuffle_4' else 10, shuffle_probability=.35)
            (folder/'config.json').write_text(json.dumps(config))
            path=folder/'metrics.json'
            metrics=json.loads(path.read_text())
            metrics.update(signature=config['signature'], optimizer_steps=200, initial_model_sha256=config['initial_model_sha256'])
            path.write_text(json.dumps(metrics))
            (folder/'train_log.jsonl').write_text('\n'.join(json.dumps(dict(epoch=e,loss=4/e,lr=3e-4*(1+np.cos(np.pi*e/200))/2,optimizer_steps=e)) for e in range(1,201)))


@pytest.mark.parametrize('mismatch', ['initial_model_sha256', 'raw_scale_training_only_g', 'train_image_ids',
                                     'validation_cache', 'source_sha256', 'optimizer_steps', 'swa_last_epochs'])
def test_rejects_unmatched_training_protocol_when_evidence_present(tmp_path, mismatch):
    from biomass_jepa.block_shuffle_report import report
    root,folds,frame=fixture_run(tmp_path)
    add_protocol_configs(root,frame)
    path=root/'arms/shuffle_4/fold_0/config.json'
    config=json.loads(path.read_text())
    if mismatch=='source_sha256':
        config['feature_origin'][mismatch]='b'*64
    elif mismatch=='swa_last_epochs':
        config['signature'][mismatch]=20
    elif mismatch=='optimizer_steps':
        mpath=path.parent/'metrics.json'
        metrics=json.loads(mpath.read_text()); metrics[mismatch]=100
        mpath.write_text(json.dumps(metrics))
    elif mismatch=='train_image_ids':
        config[mismatch]=config[mismatch][1:]
    else:
        config[mismatch]='changed'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        report(root,folds,draws=5,examples=False)


def test_real_photo_examples_use_verified_originals_and_synthetic_disclosure(tmp_path):
    from PIL import Image
    import hashlib
    from biomass_jepa.block_shuffle_report import report
    root,folds,frame=fixture_run(tmp_path,omit_c=True)
    frame['absolute_path']=''
    for i in range(2):
        photo=tmp_path/f'real_source_{i}.png'
        Image.new('RGB',(2000,1000),(10+i,40,20)).save(photo)
        frame.loc[i,'absolute_path']=str(photo)
        frame.loc[i,'image_sha256']=hashlib.sha256(photo.read_bytes()).hexdigest()
    frame.to_csv(folds,index=False)
    result=report(root,folds,draws=5,examples=True)
    assert result['figures']['figure_02_examples']['status']=='rendered'
    source=json.loads((root/'figures/figure_02_examples_source.json').read_text())
    assert len(source['sources'])==2
    assert source['sources'][0]['synthetic_columns']==[4,10]
    assert 'Synthetic' in source['caption']


def test_matched_protocol_and_real_historical_logs_are_reported(tmp_path):
    from biomass_jepa.block_shuffle_report import report
    root,folds,frame=fixture_run(tmp_path)
    add_protocol_configs(root,frame)
    ssl=tmp_path/'historical_metrics.jsonl'
    ssl.write_text('\n'.join(json.dumps(dict(step=s,loss=1/s)) for s in [100,200,300]))
    result=report(root,folds,draws=5,examples=False,ssl_log=ssl)
    assert result['protocol_audit']['status']=='verified'
    assert result['learning_dynamics']['historical_ssl']['status']=='available'
    assert result['protocol_audit']['folds']['0']['optimizer_steps']==dict(control=200,shuffle_4=200,shuffle_10=200)
    assert set(result['comparisons'])=={'shuffle_4_minus_control','shuffle_10_minus_control','shuffle_4_minus_shuffle_10'}


def test_bootstrap_samples_whole_groups_with_the_same_rng(tmp_path):
    from biomass_jepa.block_shuffle_report import paired_bootstrap, raw_weighted_r2
    y=np.array([[1,2,1,2,4],[3,4,2,5,9],[9,1,2,11,12]],float)
    p=y+np.array([[1],[2],[3]])
    groups=np.array(['a','a','b'])
    samples,_=paired_bootstrap(y,{'control':p},groups,draws=20,seed=17)
    rng=np.random.default_rng(17)
    for draw in range(20):
        chosen=rng.choice(['a','b'],2,replace=True)
        rows=np.concatenate([np.flatnonzero(groups==group) for group in chosen])
        assert samples.loc[draw,'control']==pytest.approx(raw_weighted_r2(y[rows],p[rows]))


def test_incomplete_treatment_saved_rows_do_not_produce_a_paired_result(tmp_path):
    from biomass_jepa.block_shuffle_report import report
    root,folds,_=fixture_run(tmp_path)
    path=root/'arms/shuffle_10/oof_predictions.csv'
    path.unlink()
    result=report(root,folds,draws=5,examples=False)
    assert result['arms']['shuffle_10']['status']=='incomplete'
    assert result['arms']['shuffle_10']['saved_fold_rows']==12
    assert all('shuffle_10' not in key for key in result['comparisons'])


def test_degenerate_bootstrap_has_explicit_undefined_interval():
    from biomass_jepa.block_shuffle_report import paired_bootstrap
    truth=np.ones((3,5))
    samples,comparisons=paired_bootstrap(truth,{'control':truth,'shuffle_4':truth},['a','a','b'],draws=5)
    assert comparisons['shuffle_4_minus_control']['ci95'] is None
    assert comparisons['shuffle_4_minus_control']['observed_difference'] is None
    assert samples['shuffle_4_minus_control'].isna().all()


@pytest.mark.parametrize('derived',['pred_GDM_g','pred_Dry_Total_g'])
def test_prediction_sum_identities_are_required(tmp_path,derived):
    from biomass_jepa.block_shuffle_report import _validate_predictions
    root,_,frame=fixture_run(tmp_path)
    pred=pd.read_csv(root/'arms/control/oof_predictions.csv')
    pred.loc[0,derived]+=1
    with pytest.raises(ValueError,match='identity'):
        _validate_predictions(pred,frame)


def test_ground_truth_sum_anomaly_is_preserved(tmp_path):
    from biomass_jepa.block_shuffle_report import _validate_predictions
    root,_,frame=fixture_run(tmp_path)
    pred=pd.read_csv(root/'arms/control/oof_predictions.csv')
    frame.loc[0,'Dry_Total_g']+=7
    pred.loc[0,'Dry_Total_g']+=7
    validated=_validate_predictions(pred,frame)
    assert validated.loc[0,'Dry_Total_g']==frame.loc[0,'Dry_Total_g']


@pytest.mark.parametrize('scope',['pooled','fold'])
def test_saved_global_score_must_match_independent_recalculation(tmp_path,scope):
    from biomass_jepa.block_shuffle_report import _load_arm
    root,_,frame=fixture_run(tmp_path)
    folder=root/'arms/control'
    path=folder/'metrics.json' if scope=='pooled' else folder/'fold_0/metrics.json'
    metrics=json.loads(path.read_text())
    metrics['global_weighted_r2_raw']=.12345
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError,match='score'):
        _load_arm(folder,frame)


@pytest.mark.parametrize('bad_runtime',['gpu','precision','feature_precision'])
def test_real_protocol_rejects_wrong_gpu_or_active_precision(tmp_path,bad_runtime):
    from biomass_jepa.block_shuffle_report import _verify_protocol
    root,_,frame=fixture_run(tmp_path)
    add_protocol_configs(root,frame)
    path=root/'arms/shuffle_4/fold_0/config.json'
    config=json.loads(path.read_text())
    if bad_runtime=='gpu':
        config['runtime']['gpu']='NVIDIA T4'
    elif bad_runtime=='precision':
        config['runtime']['precision']='bf16'
    else:
        config['feature_origin']['precision']='bf16'
    path.write_text(json.dumps(config))
    predictions={a:pd.read_csv(root/'arms'/a/'oof_predictions.csv') for a in ['control','shuffle_4','shuffle_10']}
    with pytest.raises(ValueError):
        _verify_protocol(root,frame,predictions)


@pytest.mark.parametrize('missing',['file','epoch','field'])
def test_missing_epoch_lr_evidence_marks_protocol_audit_incomplete(tmp_path,missing):
    from biomass_jepa.block_shuffle_report import _verify_protocol
    root,_,frame=fixture_run(tmp_path)
    add_protocol_configs(root,frame)
    path=root/'arms/shuffle_4/fold_0/train_log.jsonl'
    if missing=='file':
        path.unlink()
    else:
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if missing=='epoch':
            rows=rows[1:]
        else:
            del rows[0]['lr']
        path.write_text('\n'.join(json.dumps(row) for row in rows))
    predictions={a:pd.read_csv(root/'arms'/a/'oof_predictions.csv') for a in ['control','shuffle_4','shuffle_10']}
    audit=_verify_protocol(root,frame,predictions)
    assert audit['status']=='incomplete_provenance'
    assert audit['folds']['0']['matched_lr_log_evidence'] is False


@pytest.mark.parametrize('signature_tiny,origin_tiny',[(True,True),(True,False),(False,True)])
def test_cpu_precision_bypass_requires_explicit_tiny_source_and_fit(tmp_path,signature_tiny,origin_tiny):
    from biomass_jepa.block_shuffle_report import _verify_protocol
    root,_,frame=fixture_run(tmp_path)
    add_protocol_configs(root,frame)
    for path in (root/'arms').glob('*/fold_*/config.json'):
        config=json.loads(path.read_text())
        config['signature'].update(tiny=signature_tiny,precision='bf16')
        config['feature_origin'].update(tiny=origin_tiny,precision='bf16')
        config['runtime'].update(gpu='CPU tiny test',precision='bf16')
        path.write_text(json.dumps(config))
    predictions={a:pd.read_csv(root/'arms'/a/'oof_predictions.csv') for a in ['control','shuffle_4','shuffle_10']}
    if signature_tiny and origin_tiny:
        audit=_verify_protocol(root,frame,predictions)
        assert audit['folds']['0']['tiny_software_smoke'] is True
    else:
        with pytest.raises(ValueError):
            _verify_protocol(root,frame,predictions)


def test_tiny_report_cannot_be_presented_as_scientific_results(tmp_path):
    from biomass_jepa.block_shuffle_report import report
    root,folds,frame=fixture_run(tmp_path)
    add_protocol_configs(root,frame)
    for path in (root/'arms').glob('*/fold_*/config.json'):
        config=json.loads(path.read_text())
        config['signature'].update(tiny=True,precision='bf16')
        config['feature_origin'].update(tiny=True,precision='bf16')
        config['runtime'].update(gpu='CPU tiny test',precision='bf16')
        path.write_text(json.dumps(config))
    result=report(root,folds,draws=5,examples=False)
    assert result['cohort']['scope']=='tiny synthetic software smoke only'
    assert result['scientific_results'] is False
    assert all('no scientific inference' in row['interpretation'] for row in result['comparisons'].values())


@pytest.mark.parametrize('active,layers,accepted',[('fp8_linear_bf16_other',4,True),('bf16',4,False),('fp8_linear_bf16_other',0,False)])
def test_producer_precision_dictionary_requires_active_fp8_layers(tmp_path,active,layers,accepted):
    from biomass_jepa.block_shuffle_report import _verify_protocol
    root,_,frame=fixture_run(tmp_path)
    add_protocol_configs(root,frame)
    for path in (root/'arms').glob('*/fold_*/config.json'):
        config=json.loads(path.read_text())
        config['runtime']['precision']=dict(requested='fp8',active=active,float8_layers=layers,teacher='bf16')
        path.write_text(json.dumps(config))
    predictions={a:pd.read_csv(root/'arms'/a/'oof_predictions.csv') for a in ['control','shuffle_4','shuffle_10']}
    if accepted:
        assert _verify_protocol(root,frame,predictions)['status']=='verified'
    else:
        with pytest.raises(ValueError):
            _verify_protocol(root,frame,predictions)
