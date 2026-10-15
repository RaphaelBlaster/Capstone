"""Convolution-free depth2/4/8 comparison with mandatory in-run GPU release gates."""
import argparse
import gc
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import dino_finetune as ft
from . import train as common
from .data import TARGETS, file_sha256
from .dinov3_dense import load_dinov3
from .final_metrics import global_r2, per_target, paired_bootstrap

DEPTHS=(2,4,8)

def exact_tree(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(exact_tree(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(exact_tree(x,y) for x,y in zip(a,b))
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    return a==b

def finish(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms={};models=[];contrasts={}
    for depth in DEPTHS:
        rows=pd.concat([pd.read_csv(output/f'fold_{fold}'/f'depth_{depth}'/'val_predictions.csv') for fold in (0,1)]).sort_values('image_path').reset_index(drop=True)
        rows.to_csv(output/f'depth_{depth}_pooled_predictions.csv',index=False,float_format='%.17g')
        arms[depth]=rows;y=rows[TARGETS].to_numpy();pred=rows[['pred_'+t for t in TARGETS]].to_numpy()
        high=y[:,4]>=np.quantile(y[:,4],.8)
        models.append({'depth':depth,'global_weighted_r2_raw':global_r2(y,pred),'per_target':per_target(y,pred),'high_Total_mean_residual_g':float((pred-y)[high,4].mean())})
    reference=arms[2];keys=['image_path','Sampling_Date','fold']+TARGETS
    for depth in (4,8):
        if not reference[keys].equals(arms[depth][keys]):raise ValueError('Paired image/label identities differ.')
        contrasts[f'depth{depth}_minus_depth2']=paired_bootstrap(reference[TARGETS].to_numpy(),reference[['pred_'+t for t in TARGETS]].to_numpy(),arms[depth][['pred_'+t for t in TARGETS]].to_numpy(),reference.Sampling_Date.to_numpy())
    ft.write(output/'results.json',{'models':models,'paired_date_bootstrap':contrasts,'images':len(reference),'date_groups':reference.Sampling_Date.nunique(),'development_only':True,'reused_folds':True,'competition_score':None,'F3_enabled':False})
    pd.DataFrame([{'depth':m['depth'],'global_weighted_r2_raw':m['global_weighted_r2_raw'],'high_Total_mean_residual_g':m['high_Total_mean_residual_g']} for m in models]).to_csv(output/'depth_scores.csv',index=False)
    fig,ax=plt.subplots();ax.plot([m['depth'] for m in models],[m['global_weighted_r2_raw'] for m in models],marker='o');ax.set(xlabel='Trainable final transformer blocks (+ normalization)',ylabel='Pooled development raw weighted R²',xticks=list(DEPTHS));fig.tight_layout();fig.savefig(output/'depth_comparison.png');plt.close(fig)
    fig,ax=plt.subplots()
    for fold in (0,1):
        for depth in DEPTHS:
            rows=pd.read_csv(output/f'fold_{fold}'/f'depth_{depth}'/'training.csv');ax.plot(rows.epoch,rows.validation_global_r2,label=f'Depth{depth} fold{fold}')
    ax.set(xlabel='Epoch',ylabel='Development raw weighted R²');ax.legend();fig.tight_layout();fig.savefig(output/'depth_training_curves.png');plt.close(fig)
    ft.write(output/'FILES_SHA256.json',{str(p.relative_to(output)):file_sha256(p) for p in output.rglob('*') if p.is_file() and p.name not in ('FILES_SHA256.json','STATUS.json')})
    ft.write(output/'STATUS.json',{'status':'completed_pending_independent_rescoring','depths':list(DEPTHS),'F3_enabled':False})

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['source','images','repository','checkpoint','metadata','lock','output','verification','authorization','embedded-package-sha']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--execution-budget-seconds',type=float,default=21000)
    args=parser.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    review=json.loads(Path(args.verification).read_text());auth=json.loads(Path(args.authorization).read_text())
    ft.require_execution_gate('full',review,auth,args.embedded_package_sha)
    if review.get('depths')!=list(DEPTHS) or auth.get('depths')!=list(DEPTHS):raise ValueError('Reviewed/authorized depths differ.')
    lock=json.loads(Path(args.lock).read_text());metadata=json.loads(Path(args.metadata).read_text())
    frame=ft.audit_source(args.source,args.images,lock)
    if file_sha256(args.checkpoint)!=metadata['checkpoint_sha256']:raise ValueError('Official checkpoint hash differs.')
    device=torch.device('cuda');gpu=torch.cuda.get_device_name()
    if 'rtx pro 6000' not in gpu.lower() or torch.cuda.get_device_capability()[0]<10:raise ValueError('Actual RTXPRO6000 Blackwell required.')
    print('ACTUAL GPU:',gpu,flush=True)
    ft.write(output/'CONFIG.json',{'depths':{str(d):asdict(ft.FinetuneConfig(unfreeze_blocks=d)) for d in DEPTHS},'folds':[0,1],'package_sha256':args.embedded_package_sha,'head':'unchanged_D2','no_convolution':True,'F3_enabled':False,'original_checkpoint_sha256':metadata['checkpoint_sha256'],'input_lock':lock})
    def fresh(fold):
        model,origin=load_dinov3(args.repository,args.checkpoint,metadata,device)
        rel=f'dinov3/fold_{fold}/D2/head_selected.pt';head,_=ft.load_d2_head(Path(args.source)/rel,lock['inputs'][rel],fold,device)
        return model,head,{**origin,'D2_head_sha256':lock['inputs'][rel],'embedded_package_sha256':args.embedded_package_sha}
    def release():gc.collect();torch.cuda.empty_cache()
    # Saved cache is read-only initialization oracle; NEVER used for encoder training.
    cache_path=Path(args.source)/'dinov3/features.pt'
    cache_manifest=json.loads((Path(args.source)/'dinov3/cache_manifest.json').read_text())
    if file_sha256(cache_path)!=cache_manifest['features_sha256']:raise ValueError('Read-only parity cache hash differs from pinned manifest.')
    cache=torch.load(cache_path,weights_only=True,map_location='cpu')['features'][0]
    parity=[]
    subset=pd.concat([frame[frame.fold!=0].iloc[:32],frame[frame.fold==0].iloc[:2]]).reset_index(drop=True)
    probe=ft.read_views(subset[subset.fold==0]).to(device)
    expected_online=None
    matched_initial={}
    for fold in (0,1):
        for depth in DEPTHS:
            m,h,_=fresh(fold);ft.configure_backbone(m,'F2',depth);precision=ft.configure_precision(m,h,device)
            ids=np.flatnonzero(frame.fold.to_numpy()==fold)
            with torch.no_grad(),common.amp(device):pred=h(cache[ids].to(device)).float().cpu().numpy();online=ft.forward_dense(m,probe).float().cpu()
            saved=pd.read_csv(Path(args.source)/f'dinov3/fold_{fold}/D2/val_predictions.csv').set_index('image_path').loc[frame.iloc[ids].image_path]
            difference=float(np.abs(pred-saved[['pred_'+t for t in TARGETS]].to_numpy()).max())
            if difference>.01:raise ValueError('D2 initialization prediction parity fails original0.01g bound.')
            if expected_online is None:expected_online=online
            if not torch.equal(expected_online,online):raise ValueError('Depth changes starting online patch features.')
            # FP8 dynamic scaling depends on the batch: reproduce original adjacent
            # photograph pairs when comparing to that cache, not a new heldout pair.
            cache_errors=[]
            for idx in ids[:2]:
                begin=int(idx)//2*2;rows=frame.iloc[begin:begin+2]
                with torch.no_grad(),common.amp(device):original_batch=ft.forward_dense(m,ft.read_views(rows).to(device)).float().cpu()
                cache_errors.append(float((original_batch[int(idx)-begin]-cache[idx]).abs().max()))
            online_cache_error=max(cache_errors)
            if online_cache_error!=0.:raise ValueError('Online512 dense grid differs from saved FP8 oracle.')
            parity.append({'fold':fold,'depth':depth,'max_prediction_difference_g':difference,'online_dense_cache_max_abs':online_cache_error,'precision':precision})
            del m,h;release()
    del cache,probe;release()
    ft.write(output/'INITIALIZATION_PARITY.json',parity)
    checks=[];estimate=0.
    for depth in DEPTHS:
        config=ft.FinetuneConfig(epochs=1,unfreeze_blocks=depth)
        folder=output/'gpu_smoke'/f'depth_{depth}';torch.cuda.reset_peak_memory_stats();t=time.monotonic()
        # Real training image path, 32 originals -> two optimizer windows, same signatures.
        m,h,origin=fresh(0);a=ft.fit_arm(subset,m,h,0,'F2',folder/'uninterrupted',config=config,origin=origin);del m,h;release()
        cost=time.monotonic()-t;peak=torch.cuda.max_memory_allocated();t=time.monotonic()
        m,h,origin=fresh(0);stopped=ft.fit_arm(subset,m,h,0,'F2',folder/'resumed',config=config,origin=origin,stop_after_updates=1)
        if stopped['complete'] or stopped['optimizer_steps']!=1:raise ValueError('Smoke interruption cursor differs.')
        del m,h;release()
        m,h,origin=fresh(0);b=ft.fit_arm(subset,m,h,0,'F2',folder/'resumed',config=config,origin=origin);del m,h;release()
        left=torch.load(folder/'uninterrupted/latest.pt',weights_only=True,map_location='cpu');right=torch.load(folder/'resumed/latest.pt',weights_only=True,map_location='cpu')
        for key in ['backbone_delta','head','optimizer','rng','steps','epoch','window_cursor']:
            if not exact_tree(left[key],right[key]):raise ValueError(f'Depth{depth} exact GPU resume failed: {key}')
        p=pd.read_csv(folder/'uninterrupted/val_predictions.csv');q=pd.read_csv(folder/'resumed/val_predictions.csv')
        if not p.equals(q):raise ValueError('Exact resumed validation predictions differ.')
        if a['schedule_sha256']!=b['schedule_sha256'] or a['initial_head_sha256']!=b['initial_head_sha256']:raise ValueError('Smoke initial state/schedule differs.')
        updates=sum(sum(len(e) for e in ft.make_schedule(frame,f,20,2,8,42)) for f in (0,1))
        projected=cost/2*updates*1.5;estimate+=projected
        checks.append({'depth':depth,'exact_resume':True,'actual_updates':a['updated_backbone_names'],'frozen_unchanged':a['frozen_unchanged'],'gradient_max_abs':a['gradient_max_abs'],'trainable_parameters':sum(v.numel() for v in left['backbone_delta'].values()),'smoke_uninterrupted_seconds':cost,'smoke_resume_seconds':time.monotonic()-t,'peak_allocated_bytes':peak,'full_updates':updates,'conservative_full_seconds':projected,'selected_sha256':a['selected_file_sha256'],'precision':a['precision']})
        del left,right;release()
        print('DEPTH GPU GATE PASS:',json.dumps(checks[-1]),flush=True)
    elapsed=time.monotonic()-start
    gate={'status':'PASS','depths':checks,'initialization_parity':parity,'package_sha256':args.embedded_package_sha,'actual_gpu':gpu,'estimated_full_seconds_with_50pct_margin':estimate,'elapsed_setup_smoke_seconds':elapsed,'execution_budget_seconds':args.execution_budget_seconds}
    if elapsed+estimate>args.execution_budget_seconds:raise ValueError('Measured full runtime exceeds fixed execution budget; no full training.')
    ft.write(output/'GPU_RELEASE_GATE.json',gate);print('ALL DEPTH GPU GATES PASS; STARTING FULL MATCHED TRAINING',flush=True)
    # Bounded dispatch timeout also applies externally; never write into old studies.
    for fold in (0,1):
        for depth in DEPTHS:
            if time.monotonic()-start>=args.execution_budget_seconds:raise TimeoutError('Fixed depth-study runtime budget exhausted.')
            m,h,origin=fresh(fold);result=ft.fit_arm(frame,m,h,fold,'F2',output/f'fold_{fold}'/f'depth_{depth}',config=ft.FinetuneConfig(unfreeze_blocks=depth),origin=origin)
            identity={k:result['signature'][k] for k in ['schedule_sha256','table_sha256','initial_head_sha256','initial_backbone_sha256']}
            if fold in matched_initial and matched_initial[fold]!=identity:raise ValueError('Full depth initialization/exposure mismatch.')
            matched_initial[fold]=identity
            del m,h;release()
    finish(output)

if __name__=='__main__':main()
