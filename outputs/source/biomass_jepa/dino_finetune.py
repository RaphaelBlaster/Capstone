"""Matched D2 continuation from images. Preparation is the default; no dispatch API.

F1 freezes the encoder; F2 trains only blocks 10/11 and patch normalization.
The historical extractor/head implementation is unchanged. F3 is preparation-only.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd
# Set before CUDA/cuBLAS initialization; required by strict deterministic matmul.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm
from . import train as common
from .data import TARGETS, file_sha256, image_table
from .dinov3_dense import DINODensityHead, dense_grid, patch_tokens, head_loss, load_dinov3
from .final_metrics import global_r2, per_target, paired_bootstrap
from .predictor_specialize import state_digest
from .within_image_jigsaw import native_views, preprocess

@dataclass(frozen=True)
class FinetuneConfig:
    epochs: int = 20
    unfreeze_blocks: int = 2
    batch_size: int = 2
    accumulate: int = 8
    head_lr: float = 3e-5
    backbone_lr: float = 2e-6
    head_weight_decay: float = .01
    backbone_weight_decay: float = 0.
    l2sp_coefficient: float = 1e6
    clip_norm: float = 1.
    seed: int = 42
    endpoint: str = 'fixed_final_epoch'
    input_size: int = 512
    augmentation: str = 'identity_matches_executed_D2'
    def __post_init__(self):
        if type(self.unfreeze_blocks) is not int or not 1<=self.unfreeze_blocks<=12:
            raise ValueError('Unfreezing depth must be an integer between1 and12.')
        if self.epochs<1 or self.batch_size<1 or self.accumulate<1 or self.input_size!=512:
            raise ValueError('Positive schedule and locked512 input required.')
        if self.endpoint!='fixed_final_epoch' or self.augmentation!='identity_matches_executed_D2':
            raise ValueError('Continuation endpoint/augmentation differs from locked protocol.')
        if self.l2sp_coefficient<=0 or self.head_lr<=0 or self.backbone_lr<=0:
            raise ValueError('Positive locked regularization and learning rates required.')


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def finite(v):
        if isinstance(v,float) and not math.isfinite(v):return None
        if isinstance(v,dict):return {k:finite(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [finite(x) for x in v]
        return v
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(finite(value),indent=2,allow_nan=False));tmp.replace(path)


def configure_backbone(model,arm,unfreeze_blocks=2):
    if type(unfreeze_blocks) is not int or not 1<=unfreeze_blocks<=12:
        raise ValueError('Unfreezing depth must be an integer between1 and12.')
    if arm not in ('F1','F2','F3') or len(model.blocks)!=12:
        raise ValueError('Twelve-block DINO and F1/F2/F3 required.')
    model.requires_grad_(False).eval()
    if arm!='F1':
        for block in model.blocks[-unfreeze_blocks:]:block.requires_grad_(True)
        model.norm.requires_grad_(True)
    # Authentic B16 has wholly masked, zero qkv biases; these cannot learn.
    for module in model.modules():
        if hasattr(module,'bias_mask'):
            if module.bias is None or not torch.equal(module.bias_mask,torch.zeros_like(module.bias_mask)) or torch.count_nonzero(module.bias.detach()):
                raise ValueError('Unsupported masked attention bias state; locked B16 zero state required.')
            module.bias.requires_grad_(False)
    # eval preserves official D2 rotary-position behavior; eval DOES NOT disable gradients.
    return {n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}



def masked_bias_contract(model,restore=None):
    """Locked B16 dormant masks survive generic FP8 replacement without changing keys.

    This is valid ONLY for exactly zero masks and exactly zero frozen biases.
    Nonzero/mixed masked states require a different converter and are rejected.
    """
    if restore is not None:
        for name,mask in restore.items():
            module=model.get_submodule(name)
            if not hasattr(module,'bias_mask'):module.register_buffer('bias_mask',mask.to(module.weight.device))
    result={}
    for name,module in model.named_modules():
        if hasattr(module,'bias_mask'):
            mask=module.bias_mask;bias=module.bias
            if bias is None or bias.requires_grad or mask.shape!=bias.shape or torch.count_nonzero(mask) or torch.count_nonzero(bias.detach()):
                raise ValueError('Locked B16 attention masks/bias must stay exactly zero and frozen.')
            result[name]=mask.detach().clone()
    if restore is not None and set(result)!=set(restore):raise ValueError('FP8 conversion changed masked attention module identities.')
    return result


def configure_precision(model,head,device):
    from .final_precision import configure
    if torch.device(device).type=='cuda':
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
            raise ValueError('Deterministic CUDA requires a supported cuBLAS workspace configuration.')
        # Flash-attention backward otherwise differs even with identical inputs/RNG.
        # Preserve FP8/BF16 dtypes and fail rather than permit nondeterministic kernels.
        torch.use_deterministic_algorithms(True)
    masks=masked_bias_contract(model)
    # Generic FP8 linear omits masked-bias multiplication, exactly equivalent only
    # because this verified checkpoint's bias is frozen at zero. Restore buffers.
    backbone=configure(model,device);masked_bias_contract(model,masks)
    return {'backbone':backbone,'head':configure(head,device),'masked_bias':'exact_zero_frozen_buffers_preserved'}


def forward_dense(model,images):
    if images.ndim!=5 or tuple(images.shape[1:])!=(2,3,512,512):
        raise ValueError('Two complete512 RGB views per original required.')
    tokens=patch_tokens(model.forward_features(images.flatten(0,1)),len(images)*2)
    return dense_grid(tokens,len(images))


def anchored_penalty(model,anchors):
    parameters=dict(model.named_parameters())
    if not anchors:return next(model.parameters()).new_zeros(())
    if set(anchors)!={n for n,p in parameters.items() if p.requires_grad}:
        raise ValueError('Fixed anchors must cover exactly intended trainable backbone weights.')
    if any(a.requires_grad or a.shape!=parameters[n].shape for n,a in anchors.items()):
        raise ValueError('Detached immutable original anchors required.')
    count=sum(a.numel() for a in anchors.values())
    return sum((parameters[n].float()-a.float()).square().sum() for n,a in anchors.items())/count


def make_optimizer(model,head,config,arm):
    groups=[{'params':list(head.parameters()),'lr':config.head_lr,'weight_decay':config.head_weight_decay,'group_name':'D2_head'}]
    trainable=[p for p in model.parameters() if p.requires_grad]
    if arm!='F1':groups.append({'params':trainable,'lr':config.backbone_lr,'weight_decay':config.backbone_weight_decay,'group_name':f'last{config.unfreeze_blocks}blocks_and_norm'})
    elif trainable:raise ValueError('F1 must freeze complete backbone.')
    return torch.optim.AdamW(groups,betas=(.9,.999),eps=1e-8)


def make_schedule(frame,fold,epochs=20,batch_size=2,accumulate=8,seed=42):
    if not frame.index.equals(pd.RangeIndex(len(frame))) or frame.image_path.duplicated().any():
        raise ValueError('Unique images and consecutive index required.')
    if frame.groupby('Sampling_Date').fold.nunique().max()!=1:
        raise ValueError('Acquisition date leakage across folds.')
    train=np.flatnonzero(frame.fold.to_numpy()!=fold)
    if not len(train) or not (frame.fold==fold).any():raise ValueError('Empty train/heldout fold.')
    g=torch.Generator().manual_seed(seed+int(fold));schedule=[]
    for _ in range(epochs):
        order=train[torch.randperm(len(train),generator=g).numpy()].tolist()
        micro=[order[i:i+batch_size] for i in range(0,len(order),batch_size)]
        schedule.append([micro[i:i+accumulate] for i in range(0,len(micro),accumulate)])
    return schedule


def read_views(rows):
    return torch.stack([torch.stack([preprocess(tile,512) for tile in native_views(p)]) for p in rows.absolute_path])


def load_d2_head(path,expected_hash,fold,device='cpu'):
    if file_sha256(path)!=expected_hash:raise ValueError('D2 checkpoint SHA differs.')
    ck=torch.load(path,weights_only=True,map_location='cpu')
    if ck['signature']['arm']!='D2' or ck['signature']['fold']!=fold or ck['completed_epochs']!=200 or ck['swa_count']!=40:
        raise ValueError('Expected verified fold-specific D2 fixed40SWA initialization.')
    head=DINODensityHead(ck['scale']);head.load_state_dict(ck['model'],strict=True)
    return head.to(device),ck


def _snapshot(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters()}


def _delta_state(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}


def _load_delta(model,delta):
    params=dict(model.named_parameters())
    if set(delta)!={n for n,p in params.items() if p.requires_grad}:raise ValueError('Delta checkpoint scope differs.')
    with torch.no_grad():
        for n,v in delta.items():
            if v.shape!=params[n].shape:raise ValueError('Delta shape differs.')
            params[n].copy_(v)


def fit_arm(frame,model,head,fold,arm,output,loader=read_views,config=None,origin=None,stop_after_updates=None):
    if arm not in ('F1','F2'):raise ValueError('F3 is preparation-only; no execution is implemented.')
    config=config or FinetuneConfig();origin=origin or {};output=Path(output);output.mkdir(parents=True,exist_ok=True)
    device=next(model.parameters()).device;common.seed_all(config.seed+int(fold))
    anchors=configure_backbone(model,arm,config.unfreeze_blocks);before=_snapshot(model);initial_head=state_digest(head)
    schedule=make_schedule(frame,fold,config.epochs,config.batch_size,config.accumulate,config.seed)
    schedule_hash=hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
    table_hash=hashlib.sha256(frame[['image_path','Sampling_Date','fold','image_sha256']+TARGETS].to_csv(index=False,float_format='%.17g').encode()).hexdigest()
    signature={'config':asdict(config),'origin':origin,'fold':fold,'arm':arm,'schedule_sha256':schedule_hash,'table_sha256':table_hash,'initial_head_sha256':initial_head,'initial_backbone_sha256':state_digest(model),'code_sha256':file_sha256(__file__)}
    precision={'active':'CPU functional only','master_weights':'FP32'}
    if device.type=='cuda':
        precision=configure_precision(model,head,device)
    signature['precision']=precision
    optimizer=make_optimizer(model,head,config,arm);steps=epoch=window_cursor=0;history=[];epoch_loss=0.;epoch_seen=0;gradient_evidence={};started=time.monotonic()
    latest=output/'latest.pt'
    if latest.exists():
        ck=torch.load(latest,weights_only=True,map_location='cpu')
        if ck['signature']!=signature:raise ValueError('Resume signature differs; preserve existing directory.')
        _load_delta(model,ck['backbone_delta']);head.load_state_dict(ck['head'],strict=True);optimizer.load_state_dict(ck['optimizer']);common.restore_rng(ck['rng'])
        epoch,window_cursor,steps,history,epoch_loss,epoch_seen,gradient_evidence=[ck[k] for k in ['epoch','window_cursor','steps','history','epoch_loss','epoch_seen','gradient_evidence']]
    elif any(output.iterdir()):raise ValueError('New experiment directory contains unrecognized files.')
    def commit():
        common.atomic_save({'signature':signature,'backbone_delta':_delta_state(model),'head':head.state_dict(),'optimizer':optimizer.state_dict(),'rng':common.rng_state(),'epoch':epoch,'window_cursor':window_cursor,'steps':steps,'history':history,'epoch_loss':epoch_loss,'epoch_seen':epoch_seen,'gradient_evidence':gradient_evidence},latest)
    if not latest.exists():
        common.atomic_save({'signature':signature,'backbone_delta':_delta_state(model),'head':head.state_dict(),'anchors':{n:a.cpu() for n,a in anchors.items()}},output/'initial.pt')
        write(output/'schedule.json',schedule);commit()
    y=torch.tensor(frame[TARGETS].to_numpy(),dtype=torch.float32,device=device)
    trainable=list(head.parameters())+[p for p in model.parameters() if p.requires_grad]
    total=sum(len(e) for e in schedule);bar=tqdm(total=total,initial=steps,desc=f'fold{fold} {arm} optimizer updates')
    try:
        while epoch<config.epochs:
            model.eval();head.train()
            for g in optimizer.param_groups:
                base=config.head_lr if g['group_name']=='D2_head' else config.backbone_lr
                g['lr']=base*.5*(1+math.cos(math.pi*epoch/config.epochs))
            for wi in range(window_cursor,len(schedule[epoch])):
                window=schedule[epoch][wi];count=sum(len(ids) for ids in window);optimizer.zero_grad(set_to_none=True);window_loss=0.
                for ids in window:
                    x=loader(frame.iloc[ids]).to(device)
                    with common.amp(device):
                        dense=forward_dense(model,x);loss,parts=head_loss(head,dense,y[ids],'D2')
                    if arm=='F2' and not dense.requires_grad:raise ValueError('F2 patch feature gradient path detached.')
                    if not torch.isfinite(loss):raise ValueError('Nonfinite D2 loss.')
                    (loss*len(ids)/count).backward();window_loss+=float(loss.detach())*len(ids)
                # Once per optimizer update; correctly handles partial final windows.
                sp=anchored_penalty(model,anchors)
                if anchors:(config.l2sp_coefficient*sp).backward()
                for n,p in model.named_parameters():
                    if p.requires_grad:
                        if p.grad is None or not torch.isfinite(p.grad).all():raise ValueError('Missing/nonfinite intended backbone gradient: '+n)
                        gradient_evidence[n]=max(gradient_evidence.get(n,0.),float(p.grad.abs().max()))
                    elif p.grad is not None:raise ValueError('Frozen parameter received gradient: '+n)
                torch.nn.utils.clip_grad_norm_(trainable,config.clip_norm,error_if_nonfinite=True);optimizer.step();steps+=1;window_cursor=wi+1;epoch_loss+=window_loss;epoch_seen+=count
                commit();bar.update(1);bar.set_postfix(loss=f'{window_loss/count:.5f}',L2SP=f'{float(sp.detach()):.3g}')
                if stop_after_updates is not None and steps>=stop_after_updates:return {'complete':False,'optimizer_steps':steps}
            validation=frame[frame.fold==fold];pred,coarse=_predict(model,head,validation,loader,config.batch_size)
            truth=validation[TARGETS].to_numpy(np.float64)
            history.append({'epoch':epoch+1,'D2_training_loss':epoch_loss/epoch_seen,'l2sp':float(anchored_penalty(model,anchors).detach()),'validation_global_r2':global_r2(truth,pred),'optimizer_steps':steps,'elapsed_seconds':time.monotonic()-started})
            epoch+=1;window_cursor=0;epoch_loss=0.;epoch_seen=0;commit();pd.DataFrame(history).to_csv(output/'training.csv',index=False)
    finally:bar.close()
    changed=sorted(n for n,p in model.named_parameters() if not torch.equal(before[n],p.detach().cpu()))
    intended=sorted(anchors);frozen_unchanged=not(set(changed)-set(intended))
    if not frozen_unchanged or (arm=='F2' and (changed!=intended or not all(gradient_evidence.get(n,0)>0 for n in intended))):
        raise ValueError('Actual update/frozen parameter proof failed.')
    head_changed=state_digest(head)!=initial_head
    if not head_changed:raise ValueError('Regression head did not update.')
    selected={'signature':signature,'backbone_delta':_delta_state(model),'head':head.state_dict(),'completed_epochs':epoch,'optimizer_steps':steps,'selection':config.endpoint}
    common.atomic_save(selected,output/'selected.pt')
    # Strict serialized reload precedes all reported predictions.
    reloaded=torch.load(output/'selected.pt',weights_only=True,map_location='cpu');_load_delta(model,reloaded['backbone_delta']);head.load_state_dict(reloaded['head'],strict=True)
    validation=frame[frame.fold==fold];pred,coarse=_predict(model,head,validation,loader,config.batch_size)
    rows=validation[['image_path','Sampling_Date','fold']+TARGETS].copy()
    for j,t in enumerate(TARGETS):rows['pred_'+t]=pred[:,j];rows['coarse_'+t]=coarse[:,j]
    rows.to_csv(output/'val_predictions.csv',index=False,float_format='%.17g');saved=pd.read_csv(output/'val_predictions.csv');truth=saved[TARGETS].to_numpy(np.float64);prediction=saved[['pred_'+t for t in TARGETS]].to_numpy(np.float64)
    from .metrics import scores
    score=global_r2(truth,prediction);parity=abs(score-scores(truth,prediction)['global_weighted_r2_raw'])
    if parity>1e-10:raise ValueError('Independent metric parity failed.')
    result={'complete':True,'signature':signature,'optimizer_steps':steps,'updated_backbone_names':changed,'gradient_max_abs':gradient_evidence,'head_changed':head_changed,'frozen_unchanged':frozen_unchanged,'parameter_hashes':{n:{'initial':hashlib.sha256(before[n].contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),'final':hashlib.sha256(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()} for n,p in model.named_parameters()},'schedule_sha256':schedule_hash,'initial_head_sha256':initial_head,'final_backbone_sha256':state_digest(model),'head_sha256':state_digest(head),'selected_file_sha256':file_sha256(output/'selected.pt'),'global_weighted_r2_raw':score,'per_target':per_target(truth,prediction),'metric_parity_error':parity,'training_images':int((frame.fold!=fold).sum()),'validation_images':len(validation),'elapsed_seconds':time.monotonic()-started,'precision':precision}
    write(output/'metrics.json',result);return result


@torch.no_grad()
def _predict(model,head,frame,loader,batch_size):
    model.eval();head.eval();device=next(model.parameters()).device;fine=[];coarse=[]
    for i in range(0,len(frame),batch_size):
        with common.amp(device):
            dense=forward_dense(model,loader(frame.iloc[i:i+batch_size]).to(device));p,c=head.predictions_at_scales(dense)
        fine.append(p.float().cpu().numpy());coarse.append(c.float().cpu().numpy())
    return np.concatenate(fine),np.concatenate(coarse)


def composition_fractions(frame):
    cols=['dry_grass','dry_clover','dry_weeds','dry_total'];v=frame[cols].to_numpy(np.float64)
    if not np.isfinite(v).all() or (v[:,:3]<0).any() or (v[:,3]<=0).any():raise ValueError('Finite nonnegative masses and positive measured total required.')
    summed=v[:,:3].sum(1)
    if (summed<=0).any() or not np.allclose(summed,v[:,3],rtol=1e-6,atol=1e-6):raise ValueError('Composition denominator differs from measured dry total.')
    return v[:,:3]/summed[:,None]


def whole_photo(path):
    from PIL import Image,ImageOps
    with Image.open(path) as raw:im=ImageOps.exif_transpose(raw).convert('RGB')
    # Full image resizing retains every labeled pixel; no biomass label assigned to a crop.
    return preprocess(im,512)


def composition_loss(model,head,images,fractions):
    if images.ndim!=4 or tuple(images.shape[1:])!=(3,512,512) or fractions.shape!=(len(images),3):raise ValueError('Whole512 auxiliary photographs and three fractions required.')
    if not torch.isfinite(fractions).all() or (fractions<0).any() or not torch.allclose(fractions.sum(1),fractions.new_ones(len(images)),atol=1e-5):raise ValueError('Valid normalized composition labels required.')
    tokens=patch_tokens(model.forward_features(images),len(images));logits=head(tokens.float().mean(1))
    return -(fractions.float()*F.log_softmax(logits.float(),dim=-1)).sum(1).mean()



def auxiliary_gradient_gate(biomass_loss,auxiliary_loss,parameters,coefficient=.1,max_ratio=10.):
    params=tuple(parameters)
    bio=torch.autograd.grad(biomass_loss,params,retain_graph=True,allow_unused=True)
    aux=torch.autograd.grad(auxiliary_loss,params,retain_graph=True,allow_unused=True)
    if not bio or any(g is None or not torch.isfinite(g).all() for g in bio+aux):
        raise ValueError('Absent/nonfinite shared auxiliary gradient.')
    b=float(sum(g.float().square().sum() for g in bio).sqrt());a=float(sum(g.float().square().sum() for g in aux).sqrt())
    if b<=0 or a<=0 or coefficient<=0 or coefficient*a/b>max_ratio:
        raise ValueError('Zero or overwhelming auxiliary gradient; F3 blocked.')
    return {'passes':True,'weighted_aux_to_biomass_norm_ratio':coefficient*a/b,'auxiliary_coefficient':coefficient,'max_ratio':max_ratio}


def prepare_composition(csv,image_root,output,expected_rows=261):
    frame=pd.read_csv(csv,sep=';',encoding='utf-8-sig');fractions=composition_fractions(frame)
    if len(frame)!=expected_rows or frame.image_file_name.nunique()!=expected_rows:raise ValueError('Expected unique genuine measured composition observations.')
    missing=[];images=[];root=Path(image_root)
    for name,fraction in zip(frame.image_file_name,fractions):
        if Path(name).name!=name:raise ValueError('Auxiliary image basename required.')
        matches=list(root.rglob(name))
        if len(matches)!=1:missing.append(name);continue
        images.append({'image_file_name':name,'path':str(matches[0]),'sha256':file_sha256(matches[0]),'composition':fraction.tolist()})
    manifest={'label_csv_sha256':file_sha256(csv),'valid_labels':len(frame),'targets':['grass_fraction','clover_fraction','weed_fraction'],'missing_images':missing,'available_images':images,'auxiliary_coefficient':.1,'loss':'soft-label cross entropy of full-photograph mean normalized patch tokens, separate linear768to3 head','execution_enabled':False,'F3_status':'PREPARED_NOT_EXECUTABLE','requirements':['all image identities and usage rights checked','nonfinite/nonzero/nonoverwhelming gradient gate on training-only data','same CSIRO schedule and initialization as F2','one auxiliary microbatch per CSIRO optimizer update; auxiliary count and cost separate','independent P0 and new user authorization']}
    write(output,manifest);return manifest


def require_execution_gate(mode,review,authorization,package_sha):
    if mode=='prepare':return
    if not package_sha or not review or review.get('status')!='PASS' or review.get('package_sha256')!=package_sha:
        raise ValueError('Exact-package independent PASS required before GPU.')
    if not authorization or authorization.get('authorized') is not True or authorization.get('mode')!=mode or authorization.get('package_sha256')!=package_sha:
        raise ValueError('New explicit exact-package user execution authorization required.')


def audit_source(source,images,lock):
    source=Path(source);images=Path(images)
    for relative,expected in lock['inputs'].items():
        if file_sha256(source/relative)!=expected:raise ValueError('Pinned D2 source artifact differs: '+relative)
    frame=pd.read_csv(source/'provenance/folds.csv',dtype={'Sampling_Date':str}).sort_values('image_path').reset_index(drop=True)
    if len(frame)!=357 or frame.image_path.nunique()!=357 or frame.image_sha256.nunique()!=357 or int(frame.fold.isin([0,1]).sum())!=105 or frame[frame.fold.isin([0,1])].Sampling_Date.nunique()!=9:raise ValueError('Expected original357 and105 development images/nine dates.')
    frame['absolute_path']=[str(images/p) for p in frame.image_path]
    actual=image_table(images/'train.csv',images).sort_values('image_path').reset_index(drop=True)
    if not frame.image_path.equals(actual.image_path) or not np.array_equal(frame[TARGETS].to_numpy(),actual[TARGETS].to_numpy()):raise ValueError('CSI labels differ from original saved table.')
    for row in frame.itertuples():
        if file_sha256(row.absolute_path)!=row.image_sha256:raise ValueError('Original photo bytes differ.')
    make_schedule(frame,0,1);make_schedule(frame,1,1)
    return frame


def finish(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);arms={};metrics=[]
    for arm in ['F1','F2']:
        rows=pd.concat([pd.read_csv(output/f'fold_{fold}'/arm/'val_predictions.csv') for fold in [0,1]]).sort_values('image_path').reset_index(drop=True)
        rows.to_csv(output/f'{arm}_pooled_predictions.csv',index=False,float_format='%.17g');arms[arm]=rows
        y=rows[TARGETS].to_numpy(np.float64);p=rows[['pred_'+t for t in TARGETS]].to_numpy(np.float64)
        metrics.append({'arm':arm,'global_weighted_r2_raw':global_r2(y,p),'per_target':per_target(y,p)})
    a,b=arms['F1'],arms['F2']
    if not a[['image_path','Sampling_Date','fold']+TARGETS].equals(b[['image_path','Sampling_Date','fold']+TARGETS]):raise ValueError('Paired predictions identities differ.')
    y=a[TARGETS].to_numpy();p=a[['pred_'+t for t in TARGETS]].to_numpy();q=b[['pred_'+t for t in TARGETS]].to_numpy()
    bootstrap=paired_bootstrap(y,p,q,a.Sampling_Date.to_numpy());write(output/'paired_bootstrap.json',bootstrap)
    # Predefined descriptive upper quintile; never fit a correction on held-out errors.
    high=y[:,4]>=np.quantile(y[:,4],.8)
    write(output/'results.json',{'models':metrics,'F2_minus_F1':bootstrap,'images':len(y),'reused_development_only':True,'competition_score':None,'high_Total_subset':{'threshold_g':float(np.quantile(y[:,4],.8)),'n':int(high.sum()),'F1_mean_residual_g':float((p-y)[high,4].mean()),'F2_mean_residual_g':float((q-y)[high,4].mean())}})
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    for ax,j in zip(axes,[0,1,2,4]):
        for arm,pred in [('F1',p),('F2',q)]:ax.scatter(y[:,j],pred[:,j],s=12,alpha=.55,label=arm)
        ax.set(title=TARGETS[j],xlabel='Measured grams',ylabel='Predicted grams');ax.legend()
    fig.tight_layout();fig.savefig(output/'actual_vs_predicted.png');plt.close(fig)
    fig,ax=plt.subplots()
    for arm,pred in [('F1',p),('F2',q)]:ax.scatter(y[:,4],(pred-y)[:,4],s=12,label=arm)
    ax.axhline(0,color='black');ax.set(xlabel='Measured Total grams',ylabel='Prediction minus truth grams');ax.legend();fig.savefig(output/'Total_residuals.png');plt.close(fig)
    fig,ax=plt.subplots()
    for fold in [0,1]:
        for arm in ['F1','F2']:
            history=pd.read_csv(output/f'fold_{fold}'/arm/'training.csv');ax.plot(history.epoch,history.validation_global_r2,label=f'{arm} fold{fold}')
    ax.set(xlabel='Epoch',ylabel='Development global weighted R2');ax.legend();fig.savefig(output/'training_curves.png');plt.close(fig)
    write(output/'FILES_SHA256.json',{str(p.relative_to(output)):file_sha256(p) for p in output.rglob('*') if p.is_file() and p.name not in ['FILES_SHA256.json','STATUS.json']})
    write(output/'STATUS.json',{'status':'completed_pending_independent_P1','new_performance_unverified':True})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['source','images','repository','checkpoint','metadata','lock','output']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--mode',choices=['prepare','smoke','full'],default='prepare')
    for name in ['verification','authorization','runtime-proof','embedded-package-sha']:parser.add_argument('--'+name)
    args=parser.parse_args();config=FinetuneConfig();output=Path(args.output)
    def read_optional(path):return json.loads(Path(path).read_text()) if path else None
    require_execution_gate(args.mode,read_optional(args.verification),read_optional(args.authorization),args.embedded_package_sha)
    lock=json.loads(Path(args.lock).read_text());frame=audit_source(args.source,args.images,lock);metadata=json.loads(Path(args.metadata).read_text())
    if file_sha256(args.checkpoint)!=metadata['checkpoint_sha256']:raise ValueError('Official B16 checkpoint hash differs.')
    write(output/'CONFIG.json',asdict(config));write(output/'input_lock.json',lock)
    if args.mode=='prepare':
        write(output/'STATUS.json',{'status':'prepared_not_executed','arms':['F1','F2'],'F3':'preparation_only','gpu_launched':False});return
    device=torch.device('cuda')
    if args.mode=='full':
        proof=read_optional(args.runtime_proof)
        if not proof or proof.get('status')!='PASS' or proof.get('package_sha256')!=args.embedded_package_sha or proof.get('config')!=asdict(config) or proof.get('gpu_fp8_verified') is not True:
            raise ValueError('Independent actual GPU smoke PASS required for full training.')
    # Smoke outputs never share full directories or contribute scientific predictions.
    run_config=config if args.mode=='full' else FinetuneConfig(epochs=1,batch_size=2,accumulate=1)
    for fold in [0,1]:
        runframe=frame if args.mode=='full' else pd.concat([frame[frame.fold!=fold].iloc[:2],frame[frame.fold==fold].iloc[:2]]).reset_index(drop=True)
        for arm in ['F1','F2']:
            # Independent reloads prevent arm contamination.
            model,origin=load_dinov3(args.repository,args.checkpoint,metadata,device)
            rel=f'dinov3/fold_{fold}/D2/head_selected.pt';head,ck=load_d2_head(Path(args.source)/rel,lock['inputs'][rel],fold,device)
            fit_arm(runframe,model,head,fold,arm,output/f'fold_{fold}'/arm,read_views,run_config,{**origin,'D2_head_sha256':lock['inputs'][rel],'embedded_package_sha256':args.embedded_package_sha})
            del model,head
            torch.cuda.empty_cache()
    if args.mode=='full':finish(output)
    else:write(output/'STATUS.json',{'status':'gpu_smoke_completed_pending_independent_verification','package_sha256':args.embedded_package_sha,'config':asdict(config),'no_scientific_score':True})

if __name__=='__main__':main()
