"""Real behavioral oracles on a test-only tiny twelve-block backbone."""
import copy
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from biomass_jepa.dinov3_dense import DINODensityHead, head_loss, extract_dense
from biomass_jepa.data import TARGETS

class TinyDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed=nn.Linear(3,8)
        self.blocks=nn.ModuleList([nn.Linear(8,8) for _ in range(12)])
        self.norm=nn.LayerNorm(8)
        self.project=nn.Linear(8,768)
    def forward_features(self,x):
        t=self.patch_embed(x.mean((-2,-1)))
        for b in self.blocks:t=t+torch.tanh(b(t))
        t=self.project(self.norm(t))
        # Spatial heterogeneity makes nonlinear cross-scale loss non-degenerate.
        pos=torch.linspace(-1,1,1024,device=x.device)[None,:,None]
        return {'x_norm_patchtokens':t[:,None,:]*(1+pos*.3)}

@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(2);torch.manual_seed(7)

@pytest.fixture
def ft():
    assert importlib.util.find_spec('biomass_jepa.dino_finetune') is not None, 'New differentiable trainer missing'
    from biomass_jepa import dino_finetune
    return dino_finetune

def images(n=2):return torch.randn(n,2,3,512,512)
def table():
    f=pd.DataFrame({'image_path':['a','b','c','d','e'],'Sampling_Date':['one','two','two','three','three'],'fold':[0,1,1,2,2],'image_sha256':[str(i) for i in range(5)]})
    for j,t in enumerate(TARGETS):f[t]=np.arange(5)*2.+j+10
    return f

def test_differentiable_path_equals_frozen_readout_at_initialization(ft):
    m=TinyDINO().eval().requires_grad_(False);x=images()
    expected=extract_dense(m,x)
    anchors=ft.configure_backbone(m,'F2')
    dense=ft.forward_dense(m,x)
    torch.testing.assert_close(dense,expected,rtol=0,atol=0)
    assert dense.requires_grad and all(not a.requires_grad for a in anchors.values())
    h=DINODensityHead(20.);loss,_=head_loss(h,dense,torch.ones(2,5)*30,'D2');loss.backward()
    for i in range(12):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in m.blocks[i].parameters()) if i>=10 else all(p.grad is None for p in m.blocks[i].parameters())
    assert all(p.grad is not None for p in m.norm.parameters())
    assert all(p.grad is None for p in m.patch_embed.parameters())

def test_l2sp_fixed_mean_anchor_gradient_and_no_zero_shrink(ft):
    m=TinyDINO();anchors=ft.configure_backbone(m,'F2');fixed=copy.deepcopy(anchors)
    assert ft.anchored_penalty(m,anchors).item()==0
    with torch.no_grad():
        for n,p in m.named_parameters():
            if n in anchors:p.add_(.125)
    loss=ft.anchored_penalty(m,anchors)
    assert loss.item()==pytest.approx(.125**2)
    loss.backward();P=sum(a.numel() for a in anchors.values())
    for n,p in m.named_parameters():
        if n in anchors:
            torch.testing.assert_close(p.grad,torch.full_like(p,2*.125/P))
            assert torch.equal(anchors[n],fixed[n])

def test_f1_frozen_and_optimizer_groups_are_explicit(ft):
    c=ft.FinetuneConfig();m=TinyDINO();h=DINODensityHead(20.);ft.configure_backbone(m,'F1')
    opt=ft.make_optimizer(m,h,c,'F1');assert len(opt.param_groups)==1
    assert not any(p.requires_grad for p in m.parameters())
    ft.configure_backbone(m,'F2');opt=ft.make_optimizer(m,h,c,'F2')
    assert [g['lr'] for g in opt.param_groups]==[3e-5,2e-6]
    assert [g['weight_decay'] for g in opt.param_groups]==[.01,0.]
    assert c.l2sp_coefficient==1e6 and c.epochs==20

def test_paired_schedule_excludes_dates_and_handles_partial_windows(ft):
    f=table();a=ft.make_schedule(f,0,epochs=2,batch_size=2,accumulate=8)
    b=ft.make_schedule(f,0,epochs=2,batch_size=2,accumulate=8)
    assert a==b and len(a)==2
    assert all(sorted(i for window in epoch for batch in window for i in batch)==[1,2,3,4] for epoch in a)
    f.loc[1,'Sampling_Date']='one'
    with pytest.raises(ValueError,match='date'):ft.make_schedule(f,0,2,2,8)

def test_actual_updates_f1_f2_and_atomic_resume(ft,tmp_path):
    f=table();x=images(5);loader=lambda rows:x[rows.index.to_numpy()]
    initial=TinyDINO();head=DINODensityHead(20.)
    c=ft.FinetuneConfig(epochs=2,batch_size=2,accumulate=2)
    proofs={}
    for arm in ['F1','F2']:
        m=copy.deepcopy(initial);h=copy.deepcopy(head)
        result=ft.fit_arm(f,m,h,0,arm,tmp_path/arm,loader,c,{'test':'tiny'})
        assert result['complete'] and result['optimizer_steps']==2
        proofs[arm]=result
        assert result['updated_backbone_names']==([] if arm=='F1' else sorted(n for n,p in m.named_parameters() if p.requires_grad))
        assert result['head_changed'] and result['frozen_unchanged']
    assert proofs['F1']['schedule_sha256']==proofs['F2']['schedule_sha256']
    assert proofs['F1']['initial_head_sha256']==proofs['F2']['initial_head_sha256']
    m=copy.deepcopy(initial);h=copy.deepcopy(head)
    r=ft.fit_arm(f,m,h,0,'F2',tmp_path/'resume',loader,c,{'test':'tiny'},stop_after_updates=1)
    assert not r['complete']
    m=copy.deepcopy(initial);h=copy.deepcopy(head)
    ft.fit_arm(f,m,h,0,'F2',tmp_path/'resume',loader,c,{'test':'tiny'})
    completed=torch.load(tmp_path/'F2/selected.pt',weights_only=True)
    resumed=torch.load(tmp_path/'resume/selected.pt',weights_only=True)
    for key in ['backbone_delta','head']:
        for n in completed[key]:assert torch.equal(completed[key][n],resumed[key][n])
    assert (tmp_path/'F2/val_predictions.csv').exists()

def test_composition_separate_targets_denominator_and_gradient(ft):
    f=pd.DataFrame({'dry_grass':[10.,0.],'dry_clover':[5.,7.],'dry_weeds':[5.,3.],'dry_total':[20.,10.]})
    fractions=ft.composition_fractions(f)
    np.testing.assert_allclose(fractions,[[.5,.25,.25],[0,.7,.3]])
    for col,val in [('dry_total',0.),('dry_clover',float('nan')),('dry_grass',-1.)]:
        bad=f.copy();bad.loc[0,col]=val
        with pytest.raises(ValueError):ft.composition_fractions(bad)
    m=TinyDINO();ft.configure_backbone(m,'F3');h=nn.Linear(768,3)
    x=torch.randn(2,3,512,512)
    loss=ft.composition_loss(m,h,x,torch.tensor(fractions,dtype=torch.float32));loss.backward()
    assert h.weight.grad.abs().max()>0 and m.blocks[-1].weight.grad.abs().max()>0
    with pytest.raises(ValueError):ft.composition_loss(m,h,x,torch.ones(2,3))

def test_no_execution_without_exact_review_and_user_authorization(ft):
    with pytest.raises(ValueError):ft.require_execution_gate('full',None,None,'abc')
    review={'status':'PASS','package_sha256':'abc'};auth={'authorized':True,'mode':'full','package_sha256':'abc'}
    ft.require_execution_gate('full',review,auth,'abc')
    with pytest.raises(ValueError):ft.require_execution_gate('full',review,{**auth,'package_sha256':'different'},'abc')

def test_partial_accumulation_is_sample_mean_not_microbatch_mean(ft,tmp_path):
    f=table();x=images(5);loader=lambda rows:x[rows.index.to_numpy()]
    init=TinyDINO();head=DINODensityHead(20.)
    results=[]
    for batch,accum in [(3,8),(4,1)]:
        m=copy.deepcopy(init);h=copy.deepcopy(head)
        ft.fit_arm(f,m,h,0,'F2',tmp_path/str(batch),loader,ft.FinetuneConfig(epochs=1,batch_size=batch,accumulate=accum),{'tiny':True})
        results.append((m,h))
    for a,b in zip(results[0][1].parameters(),results[1][1].parameters()):torch.testing.assert_close(a,b,atol=3e-7,rtol=1e-5)

def test_auxiliary_gradient_gate_blocks_absent_nonfinite_overwhelming(ft):
    p=nn.Parameter(torch.tensor([1.,2.]))
    biomass=p.square().sum();aux=p.sum();proof=ft.auxiliary_gradient_gate(biomass,aux,[p],coefficient=.1)
    assert proof['passes'] and 0<proof['weighted_aux_to_biomass_norm_ratio']<10
    with pytest.raises(ValueError):ft.auxiliary_gradient_gate(biomass,aux*10000,[p],coefficient=.1)
    with pytest.raises(ValueError):ft.auxiliary_gradient_gate(biomass,aux*float('nan'),[p],coefficient=.1)

def test_f3_prepare_keeps_labels_separate_and_missing_images_block_execution(ft,tmp_path):
    f=pd.DataFrame({'image_file_name':['a.jpg'],'dry_grass':[3.],'dry_clover':[1.],'dry_weeds':[1.],'dry_total':[5.]})
    csv=tmp_path/'labels.csv';f.to_csv(csv,index=False,sep=';')
    manifest=ft.prepare_composition(csv,tmp_path,tmp_path/'manifest.json',expected_rows=1)
    assert manifest['valid_labels']==1 and manifest['missing_images']==['a.jpg']
    assert manifest['execution_enabled'] is False and manifest['auxiliary_coefficient']==.1
    assert 'Dry_Dead_g' not in manifest['targets']

class FullyMaskedLinear(nn.Linear):
    def __init__(self,width):
        super().__init__(width,width)
        self.register_buffer('bias_mask',torch.zeros_like(self.bias))
        nn.init.zeros_(self.bias)
    def forward(self,x):return torch.nn.functional.linear(x,self.weight,self.bias*self.bias_mask)

def test_dormant_official_attention_bias_is_frozen_but_weights_learn(ft):
    m=TinyDINO();m.blocks[10]=FullyMaskedLinear(8);m.blocks[11]=FullyMaskedLinear(8)
    anchors=ft.configure_backbone(m,'F2')
    assert 'blocks.10.bias' not in anchors and 'blocks.11.bias' not in anchors
    assert not m.blocks[10].bias.requires_grad and m.blocks[10].weight.requires_grad
    masks=ft.masked_bias_contract(m)
    replacement=nn.Linear(8,8);replacement.weight=m.blocks[10].weight;replacement.bias=m.blocks[10].bias
    m.blocks[10]=replacement
    ft.masked_bias_contract(m,masks)
    assert torch.equal(m.blocks[10].bias_mask,torch.zeros(8))
    loss,_=head_loss(DINODensityHead(20.),ft.forward_dense(m,images()),torch.ones(2,5)*30,'D2');loss.backward()
    assert m.blocks[10].weight.grad.abs().max()>0 and m.blocks[10].bias.grad is None
    with torch.no_grad():m.blocks[10].bias[0]=.1
    with pytest.raises(ValueError):ft.masked_bias_contract(m)

def test_unsupported_partial_attention_mask_fails_closed(ft):
    m=TinyDINO();m.blocks[10]=FullyMaskedLinear(8);m.blocks[10].bias_mask[0]=1.
    with pytest.raises(ValueError):ft.configure_backbone(m,'F2')
