import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TARGETS=['Dry_Green_g','Dry_Dead_g','Dry_Clover_g','GDM_g','Dry_Total_g']


def supervised_fixture(tmp_path, treatment_folds=(0,1,2)):
    root=tmp_path/'supervised_one_run'
    records=[]
    for i in range(12):
        green,dead,clover=2.+i,1.+i/2,1.+i/4
        records.append(dict(image_path=f'image_{i:02d}.jpg',image_sha256=f'{i:064x}',group_id=f'date_{i//2}',
            Sampling_Date=f'date_{i//2}',fold=(i//2)%3,**dict(zip(TARGETS,[green,dead,clover,green+clover,green+clover+dead]))))
    frame=pd.DataFrame(records);folds=tmp_path/'prepared.csv';frame.to_csv(folds,index=False)
    for arm,error in [('control',2.),('shuffle_4',1.)]:
        parts=[]
        for fold in range(3):
            if arm=='shuffle_4' and fold not in treatment_folds:continue
            directory=root/'arms'/arm/f'fold_{fold}';directory.mkdir(parents=True)
            pred=frame[frame.fold==fold][['image_path','group_id','fold']+TARGETS].copy()
            for name in TARGETS[:3]:pred['pred_'+name]=pred[name]+error
            pred['pred_GDM_g']=pred.pred_Dry_Green_g+pred.pred_Dry_Clover_g
            pred['pred_Dry_Total_g']=pred.pred_GDM_g+pred.pred_Dry_Dead_g
            pred.to_csv(directory/'predictions.csv',index=False);parts.append(pred)
            signature=dict(folds=[0,1,2],seed=42,epochs=200,swa_last_epochs=40,width=512,dropout=.2,
                batch_size=16,encoder_lr=5e-6,head_lr=1e-4,last_blocks=2,precision='fp8',tiny=False,
                shuffle_grid=0 if arm=='control' else 4,shuffle_probability=0. if arm=='control' else .2,
                objective='raw_global_weighted_mse_one_training_scale',selection='fixed_final_swa_endpoint_not_best_validation_epoch',
                augmentation='online_stateless_image_epoch_v1')
            proof=dict(encoder_changed=True,head_changed=True,frozen_prefix_requires_grad_false=True,frozen_prefix_probe_unchanged=True,update=1)
            config=dict(signature=signature,source_origin=dict(source_sha256='a'*64,grass_steps=2400,representation='encoder',tiny=False),
                encoder_initial_sha256='b'*64,head_initial_sha256=f'{fold:064x}',
                raw_scale_training_only_g=30.+fold,median_training_only_g=[1.,2.,3.],
                train_image_ids=frame[frame.fold!=fold].image_path.tolist(),validation_image_ids=frame[frame.fold==fold].image_path.tolist(),
                validation_augmentation='identity_only',first_update_proof=proof,
                runtime=dict(gpu='NVIDIA RTX PRO 6000 Blackwell',encoder_learning=True,predictor_used=False,
                             precision=dict(requested='fp8',active='fp8_linear_bf16_other',float8_layers=24)))
            (directory/'config.json').write_text(json.dumps(config))
            (directory/'metrics.json').write_text(json.dumps(dict(status='completed_fixed_endpoint',completed_epochs=200,swa_epochs=40,
                optimizer_steps=200,signature=signature,first_update_proof=proof)))
            (directory/'selected.pt').write_bytes(b'synthetic-selected-delta-fixture')
            (directory/'train_log.jsonl').write_text('\n'.join(json.dumps(dict(epoch=e,loss=1/e,encoder_lr=5e-6*(1+np.cos(e*np.pi/200))/2,
                head_lr=1e-4*(1+np.cos(e*np.pi/200))/2,optimizer_steps=e)) for e in range(1,201)))
        if parts:pd.concat(parts).to_csv(root/'arms'/arm/'oof_predictions.csv',index=False)
    return root,folds,frame


def test_partial_completed_fold_pairs_are_retained_without_full_coverage_claim(tmp_path):
    from biomass_jepa.supervised_shuffle_report import report
    root,folds,_=supervised_fixture(tmp_path,treatment_folds=(0,))
    result=report(root,folds,draws=20,examples=False)
    assert result['cohort']['completed_paired_folds']==[0]
    assert result['cohort']['n_original_images']==4
    assert result['cohort']['complete_authorized_folds'] is False
    assert result['experiment']['encoder_learning'] is True
    assert result['experiment']['predictor_used'] is False
    assert result['comparisons']['shuffle_4_minus_control']['observed_difference']>0
    assert result['arms']['control']['unpaired_completed_folds']==[1,2]
    source=json.loads((root/'figures/figure_01_architecture_source.json').read_text())
    assert source['encoder_learning'] is True
    assert source['shuffle_probability']==.2
    assert 'Frozen grass' not in (root/'figures/figure_01_architecture.svg').read_text()


@pytest.mark.parametrize('change',['encoder_initial_sha256','head_initial_sha256','raw_scale_training_only_g','train_image_ids',
                                   'encoder_changed','validation_augmentation','encoder_learning','source_sha256','optimizer_steps'])
def test_supervised_pair_audit_rejects_unmatched_learning_protocol(tmp_path,change):
    from biomass_jepa.supervised_shuffle_report import report
    root,folds,_=supervised_fixture(tmp_path)
    path=root/'arms/shuffle_4/fold_0/config.json';config=json.loads(path.read_text())
    if change=='encoder_changed':config['first_update_proof'][change]=False
    elif change=='encoder_learning':config['runtime'][change]=False
    elif change=='source_sha256':config['source_origin'][change]='c'*64
    elif change=='optimizer_steps':
        path=root/'arms/shuffle_4/fold_0/metrics.json';config=json.loads(path.read_text());config[change]=100
    elif change=='train_image_ids':config[change]=config[change][1:]
    else:config[change]='bad'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):report(root,folds,draws=5,examples=False)


def test_missing_pair_has_no_invented_effect(tmp_path):
    from biomass_jepa.supervised_shuffle_report import report
    root,folds,_=supervised_fixture(tmp_path,treatment_folds=())
    result=report(root,folds,draws=5,examples=False)
    assert result['comparisons']=={}
    assert result['cohort']['completed_paired_folds']==[]
    assert result['arms']['shuffle_4']['status']=='incomplete'


def test_supervised_learning_curves_identify_both_learning_components(tmp_path):
    from biomass_jepa.supervised_shuffle_report import report
    root,folds,_=supervised_fixture(tmp_path,treatment_folds=(0,))
    result=report(root,folds,draws=5,examples=False)
    assert result['learning_dynamics']['training_type']=='supervised encoder tail and head'
    source=pd.read_csv(root/'figures/figure_06_supervised_loss_lr_source.csv')
    assert set(['encoder_lr','head_lr','loss','epoch']).issubset(source)
    svg=(root/'figures/figure_06_learning_curves.svg').read_text()
    assert 'Encoder tail + head' in svg


def test_supervised_selected_signature_must_agree_with_training_config(tmp_path):
    from biomass_jepa.supervised_shuffle_report import report
    root,folds,_=supervised_fixture(tmp_path)
    path=root/'arms/shuffle_4/fold_0/metrics.json'
    metrics=json.loads(path.read_text());metrics['signature']['epochs']=100
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError,match='signature'):report(root,folds,draws=5,examples=False)


def test_native_auxiliary_retains_zero_components_and_all_original_pixels(tmp_path):
    from PIL import Image
    from biomass_jepa.supervised_shuffle_report import prepare_native_auxiliary
    images=tmp_path/'images';images.mkdir()
    rows=[]
    for i in range(2):
        name=f'field{i}.jpg';Image.new('RGB',(1058,1024),(20+i,60,30)).save(images/name)
        rows.append(dict(image_file_name=name,dry_grass=3.+i,dry_clover=0.,dry_weeds=1.,dry_total=4.+i))
    raw=tmp_path/'native.csv';pd.DataFrame(rows).to_csv(raw,sep=';',encoding='utf-8-sig',index=False)
    out=tmp_path/'prepared.csv'
    audit=prepare_native_auxiliary(raw,images,out,set(),expected_rows=2)
    prepared=pd.read_csv(out)
    assert prepared.dry_clover.tolist()==[0.,0.]
    assert audit['rows']==2
    assert audit['scene_geometry_first']==[1058,1024]
    assert audit['exact_overlap']==0
    assert len(prepared.image_sha256.unique())==2
    with pytest.raises(ValueError,match='overlap'):
        prepare_native_auxiliary(raw,images,out,{prepared.image_sha256.iloc[0]},expected_rows=2)
