import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from biomass_jepa import train as common
from biomass_jepa.data import TARGETS, file_sha256
from biomass_jepa.official_ijepa import OfficialEncoder, OFFICIAL_MODEL, OFFICIAL_REVISION
from biomass_jepa.supervised_shuffle import OnlineBiomassImages, fit, reconstruct


def fixture(tmp_path):
    torch.set_num_threads(2); torch.manual_seed(12)
    encoder=OfficialEncoder(56,0,True)
    source=tmp_path/'source.pt'
    torch.save({'signature':{'model':OFFICIAL_MODEL,'tiny':True,'size':56},
        'initialization_kind':'official_ijepa_domain_v1','official_repository_revision':OFFICIAL_REVISION,
        'official_source_epoch':299,'domain_adaptation_status':'completed_steps','step':2400,
        'official_compact_sha256':'tiny-only','encoder':encoder.state_dict()},source)
    rows=[]
    for i in range(6):
        p=tmp_path/f'{i}.png'
        Image.fromarray(np.random.default_rng(i).integers(0,256,(80,160,3),dtype=np.uint8)).save(p)
        y=[2+i,1+i*.2,.5+i*.1];y +=[y[0]+y[2],sum(y)]
        rows.append(dict(image_path=p.name,absolute_path=str(p),image_sha256=file_sha256(p),
            group_id=str(i//2),fold=i//2,**dict(zip(TARGETS,y))))
    frame=pd.DataFrame(rows);folds=tmp_path/'folds.csv';frame.to_csv(folds,index=False)
    args=SimpleNamespace(init=str(source),folds_csv=str(folds),folds=[0,1,2],output=str(tmp_path/'direct'),
        shuffle_grid=4,shuffle_probability=.2,last_blocks=2,encoder_lr=5e-6,head_lr=1e-4,
        epochs=3,swa_last_epochs=2,batch_size=2,workers=0,max_hours=1.,tiny=True,precision='bf16',seed=42,run_fold=None)
    return frame,source,args


def test_online_pairing_pixels_and_validation(tmp_path):
    frame,_,_=fixture(tmp_path)
    control=OnlineBiomassImages(frame,56,training=True,grid=0,probability=0,seed=42,fold=0,epoch=7)
    treatment=OnlineBiomassImages(frame,56,training=True,grid=4,probability=1,seed=42,fold=0,epoch=7)
    assert control.augmentation_identity(0)['rotation']==treatment.augmentation_identity(0)['rotation']
    a=treatment[0][0];assert torch.equal(a,treatment[0][0])
    assert not torch.equal(a,control[0][0])
    first=treatment.augmentation_identity(0);treatment.epoch=8
    assert first['permutation_seeds']!=treatment.augmentation_identity(0)['permutation_seeds']
    val=OnlineBiomassImages(frame,56,training=False,grid=4,probability=1,epoch=7)
    identity=OnlineBiomassImages(frame,56,training=False)
    assert torch.equal(val[0][0],identity[0][0])
    assert val.augmentation_identity(0)['use_shuffle'] is False
    assert torch.equal(treatment[0][1],control[0][1])


def test_encoder_head_updates_delta_checkpoint_and_resume(tmp_path):
    _,source,args=fixture(tmp_path)
    common.STOP_REQUESTED=False;fit(args)
    args.output=str(tmp_path/'resumed');common.STOP_REQUESTED=True;fit(args)
    stopped=torch.load(tmp_path/'resumed/fold_0/latest.pt',weights_only=True)
    assert stopped['optimizer_steps']==1 and stopped['first_update_proof']['encoder_changed']
    assert stopped['first_update_proof']['head_changed']
    assert stopped['first_update_proof']['frozen_prefix_requires_grad_false']
    common.STOP_REQUESTED=False;fit(args)
    for fold in args.folds:
        a=torch.load(tmp_path/f'direct/fold_{fold}/selected.pt',weights_only=True)
        b=torch.load(tmp_path/f'resumed/fold_{fold}/selected.pt',weights_only=True)
        assert a['swa_epochs']==2 and a['optimizer_steps']==6
        for key in ['encoder_delta','head']:
            assert a[key].keys()==b[key].keys()
            assert all(torch.equal(v,b[key][k]) for k,v in a[key].items())
        assert not any('patch_embed' in key or 'pos_embed' in key for key in a['encoder_delta'])
        model=reconstruct(source,a);model.eval()
        frame=pd.read_csv(args.folds_csv);val=OnlineBiomassImages(frame[frame.fold==fold],56,training=False)
        with torch.no_grad():pred=model(torch.stack([val[i][0] for i in range(len(val))])).numpy().astype(np.float64)
        saved=pd.read_csv(tmp_path/f'direct/fold_{fold}/predictions.csv')
        np.testing.assert_allclose(pred,saved[['pred_'+t for t in TARGETS]],rtol=1e-6)
        initial=torch.load(source,weights_only=True)['encoder']
        observed=model.encoder.state_dict()
        assert all(torch.equal(v,observed[k]) for k,v in initial.items() if k not in a['encoder_delta'])
    assert json.loads((tmp_path/'direct/metrics.json').read_text())['status']=='completed_heads'


def test_interleaved_single_fold_and_changed_signature_rejected(tmp_path):
    _,_,args=fixture(tmp_path);args.run_fold=0
    common.STOP_REQUESTED=False;fit(args)
    assert json.loads((tmp_path/'direct/metrics.json').read_text())['status']=='incomplete_heads'
    args.run_fold=1;fit(args);args.run_fold=2;fit(args)
    assert json.loads((tmp_path/'direct/metrics.json').read_text())['status']=='completed_heads'
    args.shuffle_probability=.5
    with pytest.raises(ValueError,match='configuration'):fit(args)


def test_paired_initialization_order_and_zero_probability(tmp_path):
    _,_,args=fixture(tmp_path);args.run_fold=0;args.shuffle_probability=0.
    common.STOP_REQUESTED=False;fit(args)
    args.shuffle_grid=0;args.output=str(tmp_path/'control');fit(args)
    treatment=torch.load(tmp_path/'direct/fold_0/selected.pt',weights_only=True)
    control=torch.load(tmp_path/'control/fold_0/selected.pt',weights_only=True)
    for key in ('encoder_delta','head'):
        assert all(torch.equal(v,control[key][name]) for name,v in treatment[key].items())
    a=json.loads((tmp_path/'direct/fold_0/config.json').read_text())
    b=json.loads((tmp_path/'control/fold_0/config.json').read_text())
    for key in ('encoder_initial_sha256','head_initial_sha256','train_image_ids',
                'raw_scale_training_only_g','median_training_only_g'):
        assert a[key]==b[key]
    assert treatment['optimizer_steps']==control['optimizer_steps']


def auxiliary_fixture(tmp_path,args):
    rows=[]
    for i in range(3):
        p=tmp_path/f'aux_{i}.png'
        Image.fromarray(np.random.default_rng(100+i).integers(0,256,(64,80,3),dtype=np.uint8)).save(p)
        y=[1+i,.2+i*.1,.4+i*.1]
        rows.append(dict(image_path=p.name,absolute_path=str(p),image_sha256=file_sha256(p),
            dry_grass=y[0],dry_clover=y[1],dry_weeds=y[2],dry_total=sum(y)))
    path=tmp_path/'aux.csv';pd.DataFrame(rows).to_csv(path,index=False)
    args.aux_folds_csv=str(path);args.aux_weight=.25
    return pd.DataFrame(rows)


def test_native_auxiliary_full_view_updates_and_resume(tmp_path):
    from biomass_jepa.supervised_shuffle import NativeMassImages, NativeMassHead
    _,source,args=fixture(tmp_path);aux=auxiliary_fixture(tmp_path,args);args.run_fold=0
    dataset=NativeMassImages(aux,56,seed=42,fold=0,epoch=2)
    tiles,label,_=dataset[0]
    assert tiles.shape==(1,3,56,56) and len(label)==4
    from biomass_jepa.data import normalize_transform
    with Image.open(aux.iloc[0].absolute_path) as raw:
        full=normalize_transform(56)(raw.convert('RGB'))
    expected=torch.rot90(full,dataset.augmentation_identity(0)['rotation'],(-2,-1))
    assert torch.equal(tiles[0],expected)
    common.STOP_REQUESTED=False;fit(args)
    args.output=str(tmp_path/'resumed');common.STOP_REQUESTED=True;fit(args)
    stopped=torch.load(tmp_path/'resumed/fold_0/latest.pt',weights_only=True)
    assert stopped['first_update_proof']['aux_head_changed']
    assert stopped['aux_exposures']==2 and stopped['aux_cursor']==2
    assert len(stopped['aux_order'])==3
    common.STOP_REQUESTED=False;fit(args)
    a=torch.load(tmp_path/'direct/fold_0/selected.pt',weights_only=True)
    b=torch.load(tmp_path/'resumed/fold_0/selected.pt',weights_only=True)
    assert a['aux_exposures']==a['optimizer_steps']*args.batch_size
    for key in ('encoder_delta','head','aux_head'):
        assert all(torch.equal(v,b[key][k]) for k,v in a[key].items())
    model=reconstruct(source,a);model.eval()
    with torch.no_grad():prediction=model(tiles[None],auxiliary=True)
    assert prediction.shape==(1,4)
    torch.testing.assert_close(prediction[:,3],prediction[:,:3].sum(1))
    assert (prediction>0).all()
    config=json.loads((tmp_path/'direct/fold_0/config.json').read_text())
    assert config['aux_source']['rows']==3 and config['aux_source']['weight']==.25
    assert config['aux_source']['units']=='native supplied dry-mass units; no CSI target conversion'
    assert config['first_update_proof']['aux_head_changed']
    args.aux_weight=.5
    with pytest.raises(ValueError,match='configuration'):fit(args)


def test_auxiliary_labels_and_overlap_fail_closed(tmp_path):
    _,_,args=fixture(tmp_path);auxiliary_fixture(tmp_path,args)
    frame=pd.read_csv(args.aux_folds_csv);frame.loc[0,'dry_total']+=10
    frame.to_csv(args.aux_folds_csv,index=False)
    with pytest.raises(ValueError,match='sum'):fit(args)
