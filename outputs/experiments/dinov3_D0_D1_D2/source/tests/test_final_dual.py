"""Independent-oracle checks for locked dual study (CPU is not GPU evidence)."""
import copy
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

def test_native_jigsaw_ten_distinct_and_pixel_conservation():
 from biomass_jepa.within_image_jigsaw import permutation_bank,apply_permutation
 orders=permutation_bank('abc',0,42)
 assert len({tuple(x) for x in orders})==10
 assert all(sorted(x)==list(range(100)) and x!=list(range(100)) for x in orders)
 x=np.arange(1000000,dtype=np.uint32).reshape(1000,1000)
 a=np.stack([x%256,x//256%256,x//65536%256],-1).astype('uint8')
 b=np.asarray(apply_permutation(Image.fromarray(a),orders[0]))
 codes=lambda z:np.sort(z[...,0].astype(np.uint32)+256*z[...,1].astype(np.uint32)+65536*z[...,2].astype(np.uint32),axis=None)
 assert np.array_equal(codes(a),codes(b))
 assert not np.array_equal(a,b)
 assert orders==permutation_bank('abc',0,42)

def test_schedule_excludes_val_and_covers_all_twenty_slots():
 from biomass_jepa.predictor_specialize import exposure_schedule,audit_schedule
 df=pd.DataFrame({'image_path':['a','b'],'image_sha256':['a','b'],'fold':[0,1],'Sampling_Date':['d0','d1']})
 slots=exposure_schedule(df,0)
 assert len(slots)==40 and {s['image_path'] for s in slots}=={'b'}
 assert all(sorted(s['slot'] for s in slots if s['view']==v)==list(range(20)) for v in (0,1))
 audit_schedule(df,0,slots)
 with pytest.raises(ValueError):audit_schedule(df,0,[dict(slots[0],image_path='a')])

def test_semantic_order_sentinel_and_exact_geometry():
 from biomass_jepa.semantic_predictor_features import semantic_masks,pool_predictions
 c,t=semantic_masks(4,'cpu')
 assert c[0].shape==(4,88) and all(x.shape==(4,42) for x in t)
 assert not any(torch.isin(c[0][0],m[0]).any() for m in t)
 a=torch.empty(16,42,3)
 for mask in range(4):
  for b in range(4):a[mask*4+b].fill_(mask*100+b)
 out=pool_predictions(a,2,2)
 assert torch.equal(out[:,:,0],torch.tensor([[150.,151.],[152.,153.]]))

def test_predictor_only_actual_cpu_updates_frozen_and_resume(tmp_path):
 from biomass_jepa.official_ijepa import OfficialEncoder,make_predictor
 from biomass_jepa.predictor_specialize import optimize_schedule,state_digest
 from biomass_jepa.semantic_predictor_features import semantic_masks
 torch.set_num_threads(2);torch.manual_seed(42)
 enc=OfficialEncoder(224,0,True);teacher=copy.deepcopy(enc);pred=make_predictor(enc)
 initial=copy.deepcopy(pred.state_dict());frozen=state_digest(enc),state_digest(teacher)
 c,t=semantic_masks(2,'cpu');batches=[{'contexts':c,'targets':t,'slots':[0,1]} for _ in range(3)]
 images=torch.randn(2,3,224,224)
 load=lambda batch:images
 a=optimize_schedule(enc,pred,teacher,batches,load,tmp_path/'full',{'test':1},effective_batch=2)
 assert a['steps']==3 and a['max_abs_predictor_update']>0
 assert (state_digest(enc),state_digest(teacher))==frozen
 assert all(p.grad is None and not p.requires_grad for m in (enc,teacher) for p in m.parameters())
 pred2=make_predictor(enc);pred2.load_state_dict(initial)
 optimize_schedule(enc,pred2,teacher,batches,load,tmp_path/'resumed',{'test':1},effective_batch=2,stop_after=1)
 optimize_schedule(enc,pred2,teacher,batches,load,tmp_path/'resumed',{'test':1},effective_batch=2)
 assert state_digest(pred2)==state_digest(pred)

def test_independent_global_metric_discriminates_wrong_per_target():
 from biomass_jepa.final_metrics import global_r2
 from biomass_jepa.metrics import scores
 y=np.array([[1,40,8,9,49],[2,60,7,9,69],[3,90,1,4,94]],dtype='float64');p=y+np.array([1,2,3,4,5])
 w=np.array([.1,.1,.1,.2,.5]);mu=(y*w).sum()/len(y)
 expected=1-(((y-p)**2)*w).sum()/(((y-mu)**2)*w).sum()
 assert global_r2(y,p)==pytest.approx(expected,abs=1e-14)
 assert scores(y,p)['global_weighted_r2_raw']==pytest.approx(expected,abs=1e-10)

def test_official_teacher_target_major_actual_apply_order():
 from biomass_jepa.ijepa_reference.masks import apply_masks
 from biomass_jepa.ijepa_reference.tensors import repeat_interleave_batch
 tokens=torch.arange(2*256).reshape(2,256,1).float()
 masks=[torch.tensor([[i,i+1],[i+2,i+3]]) for i in (0,20,40,60)]
 actual=repeat_interleave_batch(apply_masks(tokens,masks),2,1)
 expected=torch.cat([torch.stack([tokens[b,m[b]] for b in range(2)]) for m in masks])
 assert torch.equal(actual,expected)

def test_masked_attention_negative_control_and_specialized_reload(tmp_path):
 from biomass_jepa.official_ijepa import OfficialEncoder,make_predictor
 from biomass_jepa.semantic_predictor_features import semantic_masks,SemanticPredictorReadout
 from biomass_jepa.predictor_specialize import optimize_schedule,state_digest
 torch.set_num_threads(2);torch.manual_seed(43)
 enc=OfficialEncoder(224,0,True);teacher=copy.deepcopy(enc);pred=make_predictor(enc)
 image=torch.randn(1,2,3,224,224);c,t=semantic_masks(2,'cpu')
 with torch.no_grad():
  masked=enc(image.flatten(0,1),c)
  posthoc=torch.stack([enc(image.flatten(0,1))[b,c[0][b]] for b in range(2)])
 assert masked.shape[1]==88 and not torch.allclose(masked,posthoc)
 before=SemanticPredictorReadout(enc,pred)(image).clone()
 batches=[{'contexts':c,'targets':t,'slots':[0,1]} for _ in range(2)]
 optimize_schedule(enc,pred,teacher,batches,lambda _:image.flatten(0,1),tmp_path,{'test':2},effective_batch=2)
 ck=torch.load(tmp_path/'predictor_specialized.pt',weights_only=True);p2=make_predictor(enc);p2.load_state_dict(ck['predictor'],strict=True)
 assert state_digest(p2)==ck['state_sha256']
 after=SemanticPredictorReadout(enc,p2)(image)
 assert not torch.equal(before,after)

def test_materialized_schedule_replayed_arrays_not_worker_seed(tmp_path):
 from biomass_jepa.predictor_specialize import materialize_schedule
 df=pd.DataFrame({'image_path':['a','b'],'image_sha256':['a','b'],'fold':[0,1],'Sampling_Date':['d0','d1']})
 one=materialize_schedule(df,0,tmp_path/'schedule.pt');torch.manual_seed(7)
 two=materialize_schedule(df,0,tmp_path/'schedule.pt')
 for a,b in zip(one['batches'],two['batches']):
  assert a['slots']==b['slots']
  assert all(torch.equal(x,y) for key in ['contexts','targets'] for x,y in zip(a[key],b[key]))

def test_new_fold_head_swa_endpoint_and_resume(tmp_path):
 from biomass_jepa.final_head import fit_cached_head
 from biomass_jepa.predictor_specialize import state_digest
 torch.set_num_threads(2);torch.manual_seed(1)
 frame=pd.DataFrame({'image_path':[f'i{i}' for i in range(8)],'fold':[0]*2+[1]*6,'Sampling_Date':['d0']*2+['d1']*6})
 names=['Dry_Green_g','Dry_Dead_g','Dry_Clover_g','GDM_g','Dry_Total_g']
 y=torch.rand(8,3)*30;y=torch.cat([y,(y[:,0]+y[:,2])[:,None],y.sum(1)[:,None]],1)
 for j,n in enumerate(names):frame[n]=y[:,j].numpy()
 features=torch.randn(4,8,2,64)
 full=fit_cached_head(frame,features,0,'J_control',tmp_path/'full',{'test':1},epochs=4,swa_last=2,batch_size=2)
 fit_cached_head(frame,features,0,'J_control',tmp_path/'resume',{'test':1},epochs=4,swa_last=2,batch_size=2,stop_after_updates=2)
 resumed=fit_cached_head(frame,features,0,'J_control',tmp_path/'resume',{'test':1},epochs=4,swa_last=2,batch_size=2)
 assert full['completed_epochs']==4 and full['swa_count']==2
 assert torch.equal(torch.load(tmp_path/'full/head_selected.pt',weights_only=True)['model']['readout.4.weight'],torch.load(tmp_path/'resume/head_selected.pt',weights_only=True)['model']['readout.4.weight'])
 assert full['global_weighted_r2_raw']==resumed['global_weighted_r2_raw']

def test_locked_builder_embeds_current_modules_and_runtime_offline(tmp_path):
 import json,ast,base64,io,zipfile
 from pathlib import Path
 from tools.build_final_dual import build
 root=Path(__file__).resolve().parents[1];build(root,tmp_path,mode='smoke')
 meta=json.loads((tmp_path/'kernel-metadata.json').read_text());nb=json.loads((tmp_path/'capstone_FINAL_DUAL_JEPA_DINO.ipynb').read_text())
 assert meta['enable_internet'] is False and meta['machine_shape']=='NvidiaRtxPro6000'
 assert meta['competition_sources']==['arc-prize-2026-arc-agi-3']
 assert 'robiulislamistaken/dinov3-vitb16' in meta['dataset_sources']
 for cell in nb['cells']:
  if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
 source=''.join(nb['cells'][1]['source']);payload=source.split("base64.b64decode('")[1].split("')")[0]
 z=zipfile.ZipFile(io.BytesIO(base64.b64decode(payload)))
 for name in ['predictor_specialize','semantic_predictor_features','dinov3_dense','final_head','final_metrics']:
  assert z.read('biomass_jepa/'+name+'.py')==(root/'biomass_jepa'/f'{name}.py').read_bytes()
 assert 'fit_heads(' not in ''.join(''.join(c['source']) for c in nb['cells'])

def test_zero_initialized_head_upstream_grad_after_first_update():
 from biomass_jepa.official_heads import GatedRegressionHead
 torch.manual_seed(40);model=GatedRegressionHead(64,torch.tensor(10.),torch.tensor([10.,10.,10.]));opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
 x=torch.randn(4,2,64);y=torch.rand(4,5)*30
 ((model(x)-y)**2).mean().backward()
 assert float(model.projection[1].weight.grad.abs().max())==0
 opt.step();opt.zero_grad();((model(x)-y)**2).mean().backward()
 assert torch.isfinite(model.projection[1].weight.grad).all() and float(model.projection[1].weight.grad.abs().max())>0

def test_actual_official_predictor_forward_target_major_batch_minor_sentinel():
 from torch import nn
 from biomass_jepa.official_ijepa import OfficialEncoder,make_predictor
 from biomass_jepa.semantic_predictor_features import semantic_masks,pool_predictions
 enc=OfficialEncoder(224,0,True);pred=make_predictor(enc);c,t=semantic_masks(4,'cpu')
 class BroadcastContextSentinel(nn.Module):
  def forward(self,x):
   y=x.clone();y[:,:,0]=x[:,0,0,None];return y
 pred.predictor_blocks=nn.ModuleList([BroadcastContextSentinel()]);pred.predictor_norm=nn.Identity()
 with torch.no_grad():
  pred.predictor_embed.weight.zero_();pred.predictor_embed.bias.zero_();pred.predictor_embed.weight[:32,:32]=torch.eye(32)
  pred.predictor_proj.weight.zero_();pred.predictor_proj.bias.zero_();pred.predictor_proj.weight[:32,:32]=torch.eye(32)
  pred.mask_token.zero_();pred.predictor_pos_embed.zero_();pred.predictor_pos_embed[0,:,1]=torch.arange(256)
 contexts=torch.zeros(4,88,64)
 for b in range(4):contexts[b,:,0]=b+1
 actual=pred(contexts,c,t)
 assert torch.equal(actual.reshape(4,4,42,64)[:,:,:,1],torch.stack([m.float() for m in t]))
 assert torch.equal(pool_predictions(actual,2,2)[:,:,0],torch.tensor([[1.,2.],[3.,4.]]))
