import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from biomass_jepa import train
from biomass_jepa.data import BiomassImages, TARGETS, file_sha256
from biomass_jepa.official_heads import cache_features, fit_heads, select_training_features
from biomass_jepa.official_ijepa import OfficialEncoder, OFFICIAL_MODEL, OFFICIAL_REVISION


def setup(tmp_path):
    torch.set_num_threads(2);torch.manual_seed(12)
    encoder=OfficialEncoder(112,0,True)
    source=tmp_path/'adapted.pt'
    torch.save({'signature':{'model':OFFICIAL_MODEL,'tiny':True,'size':112},
        'initialization_kind':'official_ijepa_domain_v1','official_repository_revision':OFFICIAL_REVISION,
        'official_source_epoch':299,'domain_adaptation_status':'completed_steps','step':2400,
        'official_compact_sha256':'tiny-functional-source','encoder':encoder.state_dict()},source)
    rows=[]
    for i in range(6):
        p=tmp_path/f'{i}.png';Image.fromarray(np.random.default_rng(i).integers(0,256,(80,160,3),dtype=np.uint8)).save(p)
        y=[2+i,1+i*.2,.5+i*.1];y +=[y[0]+y[2],sum(y)]
        rows.append({'image_path':p.name,'absolute_path':str(p),'image_sha256':file_sha256(p),
            'group_id':str(i//2),'fold':i//2,**dict(zip(TARGETS,y))})
    folds=tmp_path/'folds.csv';pd.DataFrame(rows).to_csv(folds,index=False)
    return source,folds


def extract(tmp_path,source,folds,grid):
    out=tmp_path/f'cache_{grid}'
    args=SimpleNamespace(tiny=True,precision='bf16',folds_csv=str(folds),init=str(source),
        representation='encoder',output=str(out),batch_size=2,workers=0,max_hours=1.,shuffle_grid=grid)
    train.STOP_REQUESTED=False;cache_features(args)
    return out,args


def test_loader_preserves_default_and_shuffles_before_resize(tmp_path):
    from biomass_jepa.block_shuffle import permutation_seed,shuffle_square_tile
    source,folds=setup(tmp_path);frame=pd.read_csv(folds)
    normal=BiomassImages(frame,112,views=2)
    zero=BiomassImages(frame,112,views=2,block_shuffle_grid=0)
    assert torch.equal(normal[0][0],zero[0][0])
    shuffled=BiomassImages(frame,112,views=2,block_shuffle_grid=10,block_shuffle_variant=2)
    with Image.open(frame.iloc[0].absolute_path) as image:
        expected=normal.transform(shuffle_square_tile(image.crop((0,0,80,80)),10,
            permutation_seed(frame.iloc[0].image_sha256,10,2,0)))
    assert torch.equal(shuffled[0][0][0],expected)
    assert torch.equal(shuffled[0][1],normal[0][1])
    with pytest.raises(ValueError):BiomassImages(frame,112,views=2,block_shuffle_grid=3)
    with pytest.raises(ValueError):BiomassImages(frame,112,train=True,views=2,block_shuffle_grid=4)


def test_shuffle_rng_does_not_change_global_rng():
    torch.manual_seed(2);before=torch.get_rng_state().clone()
    rng=torch.Generator().manual_seed(9)
    control=torch.zeros(4,5,2,3);shuffle=torch.ones_like(control)
    actual=select_training_features(control,shuffle,torch.tensor([0,1,2]),torch.tensor([0,1,2]),1.,rng)
    assert torch.equal(before,torch.get_rng_state())
    assert torch.equal(actual,torch.ones(3,2,3))


def test_cache_guards_and_paired_head_resume(tmp_path):
    source,folds=setup(tmp_path);control,cargs=extract(tmp_path,source,folds,0)
    shuffle,sargs=extract(tmp_path,source,folds,4)
    fine,_=extract(tmp_path,source,folds,10)
    manifests=[json.loads((c/'manifest.json').read_text()) for c in [control,shuffle,fine]]
    assert [m['signature']['shuffle_grid'] for m in manifests]==[0,4,10]
    assert all(m['signature']['shuffle_version']=='blockshuffle-v1' for m in manifests)
    before=manifests[0]['sha256'];manifest_bytes=(control/'manifest.json').read_bytes();cache_features(cargs)
    assert json.loads((control/'manifest.json').read_text())['sha256']==before
    assert (control/'manifest.json').read_bytes()==manifest_bytes
    cargs.shuffle_grid=4
    with pytest.raises(ValueError,match='configuration'):cache_features(cargs)
    fit=SimpleNamespace(tiny=True,precision='bf16',folds_csv=str(folds),cache=str(control),
        shuffle_cache=str(shuffle),shuffle_prob=.35,output=str(tmp_path/'direct'),
        folds=[0,1,2],epochs=3,swa_last_epochs=2,seed=42,lr=3e-4,batch_size=2,max_hours=1.)
    train.STOP_REQUESTED=False;fit_heads(fit)
    fit.output=str(tmp_path/'resumed');train.STOP_REQUESTED=True;fit_heads(fit)
    partial=torch.load(tmp_path/'resumed/fold_0/latest.pt',weights_only=True)
    assert 'shuffle_rng' in partial and partial['optimizer_steps']==1
    train.STOP_REQUESTED=False;fit_heads(fit)
    for fold in fit.folds:
        direct=torch.load(tmp_path/f'direct/fold_{fold}/selected.pt',weights_only=True)
        resumed=torch.load(tmp_path/f'resumed/fold_{fold}/selected.pt',weights_only=True)
        assert all(torch.equal(v,resumed['model'][k]) for k,v in direct['model'].items())
        assert direct['optimizer_steps']==6
        # Reconstruct the selected head on the identity control cache: treatment
        # validation must never use the shuffled bank, even at nonzero probability.
        from biomass_jepa.official_heads import GatedRegressionHead
        head=GatedRegressionHead(64,direct['model']['scale'],torch.ones(3))
        head.load_state_dict(direct['model']);head.eval()
        val=np.flatnonzero(pd.read_csv(folds).fold.to_numpy()==fold)
        identity=torch.load(control/'orientation_0.pt',weights_only=True)['features'][val]
        with torch.no_grad():actual=head(identity).numpy()
        expected=pd.read_csv(tmp_path/f'direct/fold_{fold}/predictions.csv')
        np.testing.assert_allclose(actual,expected[['pred_'+t for t in TARGETS]],rtol=1e-6)
        from biomass_jepa.metrics import scores
        rescore=scores(expected[TARGETS],expected[['pred_'+t for t in TARGETS]])
        saved_score=json.loads((tmp_path/f'direct/fold_{fold}/metrics.json').read_text())
        assert rescore['global_weighted_r2_raw']==pytest.approx(saved_score['global_weighted_r2_raw'],abs=1e-12)
    # With probability zero the treatment path must preserve all control RNG/state.
    fit.shuffle_prob=0.;fit.output=str(tmp_path/'p0');fit_heads(fit)
    fit.shuffle_cache=None;fit.output=str(tmp_path/'control_head');fit_heads(fit)
    for fold in fit.folds:
        a=torch.load(tmp_path/f'p0/fold_{fold}/selected.pt',weights_only=True)
        b=torch.load(tmp_path/f'control_head/fold_{fold}/selected.pt',weights_only=True)
        assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items())
        assert a['initial_model_sha256']==b['initial_model_sha256']
    # Reject a shuffled baseline and mismatched treatment signatures.
    fit.cache=str(shuffle)
    with pytest.raises(ValueError,match='control'):fit_heads(fit)
    fit.cache=str(control);fit.shuffle_cache=str(fine)
    path=fine/'manifest.json';m=json.loads(path.read_text());m['signature']['source_sha256']='wrong'
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='paired|mixed-source'):fit_heads(fit)


def test_completed_cache_cannot_readopt_changed_feature_bytes(tmp_path):
    source,folds=setup(tmp_path);control,args=extract(tmp_path,source,folds,0)
    path=control/'orientation_0.pt';saved=torch.load(path,weights_only=True)
    saved['features'][0,0,0]+=1.;torch.save(saved,path)
    with pytest.raises(ValueError,match='bytes changed'):cache_features(args)
