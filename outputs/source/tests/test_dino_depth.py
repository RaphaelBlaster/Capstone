import copy
import pytest
import torch
from test_dino_finetune import TinyDINO, images, table
from biomass_jepa import dino_finetune as ft
from biomass_jepa.dinov3_dense import DINODensityHead, head_loss

@pytest.mark.parametrize('depth',[2,4,8])
def test_depth_updates_and_exact_resume(depth,tmp_path):
    torch.set_num_threads(2);torch.manual_seed(42)
    initial=TinyDINO();head=DINODensityHead(20.);frame=table();x=images(5)
    loader=lambda rows:x[rows.index.to_numpy()]
    config=ft.FinetuneConfig(epochs=2,batch_size=2,accumulate=2,unfreeze_blocks=depth)
    model=copy.deepcopy(initial)
    anchors=ft.configure_backbone(model,'F2',depth)
    assert set(anchors)=={n for n,p in model.named_parameters() if n.startswith('norm.') or any(n.startswith(f'blocks.{i}.') for i in range(12-depth,12))}
    dense=ft.forward_dense(model,x[:2]);assert dense.requires_grad
    loss,_=head_loss(head,dense,torch.ones(2,5)*30,'D2');loss.backward()
    assert all(p.grad is not None and p.grad.abs().max()>0 for n,p in model.named_parameters() if n in anchors)
    assert all(p.grad is None for n,p in model.named_parameters() if n not in anchors)
    head=DINODensityHead(20.)
    full=ft.fit_arm(frame,copy.deepcopy(initial),copy.deepcopy(head),0,'F2',tmp_path/'full',loader,config)
    assert full['frozen_unchanged'] and set(full['updated_backbone_names'])==set(anchors)
    stopped=ft.fit_arm(frame,copy.deepcopy(initial),copy.deepcopy(head),0,'F2',tmp_path/'resume',loader,config,stop_after_updates=1)
    assert not stopped['complete']
    ft.fit_arm(frame,copy.deepcopy(initial),copy.deepcopy(head),0,'F2',tmp_path/'resume',loader,config)
    a=torch.load(tmp_path/'full/latest.pt',weights_only=True);b=torch.load(tmp_path/'resume/latest.pt',weights_only=True)
    for key in ['backbone_delta','head']:
        assert a[key].keys()==b[key].keys()
        assert all(torch.equal(a[key][n],b[key][n]) for n in a[key])
    assert a['steps']==b['steps']==2

@pytest.mark.parametrize('depth',[0,13,-1,True,2.5])
def test_invalid_depth_rejected(depth):
    with pytest.raises(ValueError):ft.FinetuneConfig(unfreeze_blocks=depth)
    with pytest.raises(ValueError):ft.configure_backbone(TinyDINO(),'F2',depth)

def test_depths_keep_training_exposure_and_settings():
    from dataclasses import asdict
    configs=[ft.FinetuneConfig(unfreeze_blocks=d) for d in [2,4,8]]
    schedules=[ft.make_schedule(table(),0,c.epochs,c.batch_size,c.accumulate,c.seed) for c in configs]
    assert schedules[0]==schedules[1]==schedules[2]
    settings=[{k:v for k,v in asdict(c).items() if k!='unfreeze_blocks'} for c in configs]
    assert settings[0]==settings[1]==settings[2]
