"""Controlled D1/D3 run from immutable completed dense features; no new SSL."""
import argparse, json, hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from .data import file_sha256, TARGETS
from .final_dual_run import write
from .final_head import fit_cached_head
from .dino_weighting import make_head, head_loss
from .final_metrics import global_r2, per_target


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--lock',required=True)
    a=p.parse_args(); source=Path(a.source); out=Path(a.output); lock=json.loads(Path(a.lock).read_text())
    if not torch.cuda.is_available() or 'rtx pro 6000' not in torch.cuda.get_device_name().lower() or torch.cuda.get_device_capability()[0]<10:
        raise RuntimeError('Actual RTX PRO6000 Blackwell required; no fallback.')
    torch.set_num_threads(2)
    for relative,digest in lock['inputs'].items():
        if file_sha256(source/relative)!=digest:raise ValueError('Pinned completed input changed: '+relative)
    write(out/'STATUS.json',{'status':'running','gpu':torch.cuda.get_device_name(),'folds':[0,1],'epochs':200,'swa':40,'competition_score':None})
    frame=pd.read_csv(source/'provenance/folds.csv',dtype={'Sampling_Date':str,'group_id':str})
    cache=torch.load(source/'dinov3/features.pt',weights_only=True,map_location='cpu')
    feats=cache['features']; manifest=json.loads((source/'dinov3/cache_manifest.json').read_text())
    if cache['cursor']!=357 or feats.shape!=(1,357,2,8,8,768) or cache['signature']!=manifest['signature']:
        raise ValueError('Incomplete/foreign DINO dense cache.')
    image_lock=cache['signature']['images']
    if frame[['image_path','image_sha256']].to_dict('records')!=image_lock:raise ValueError('Cache-image alignment failed.')
    if len(frame)!=357 or frame.image_path.nunique()!=357 or frame.groupby('Sampling_Date').fold.nunique().max()!=1:raise ValueError('Original grouped folds required.')
    if manifest['signature']['checkpoint_sha256']!='73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c':raise ValueError('Authentic B16 cache required.')
    features=feats[0].cuda(); origin={'source_version':2,'input_lock':lock,'experiment':'D3 learned normalized pooling versus D1 uniform pooling; no consistency regularizer'}
    # Actual FP8 smoke, both heads, saved optimizer state, update and reload.
    subset=pd.concat([frame[frame.fold!=0].iloc[:2],frame[frame.fold==0].iloc[:2]])
    tinyfeatures=features[torch.tensor(subset.index.to_numpy(),device='cuda')]; subset=subset.reset_index(drop=True)
    proofs={}
    for arm in ['D1','D3']:
        smoke=out/'smoke'/arm
        proofs[arm]=fit_cached_head(subset,tinyfeatures,0,arm,smoke,origin,epochs=2,swa_last=1,batch_size=2,head_factory=make_head,loss_function=head_loss)
        saved=torch.load(smoke/'head_selected.pt',weights_only=True,map_location='cuda')
        model=make_head(arm,saved['scale']).cuda()
        from .final_precision import configure
        configure(model,torch.device('cuda'));model.load_state_dict(saved['model'],strict=True);model.eval()
        from . import train as common
        with torch.no_grad(),common.amp(torch.device('cuda')): predictions=model(tinyfeatures[2:]).float().cpu().numpy()
        csv=pd.read_csv(smoke/'val_predictions.csv')
        if not np.allclose(predictions,csv[['pred_'+t for t in TARGETS]].to_numpy(),rtol=0,atol=1e-5):raise ValueError('Actual FP8 strict reload mismatch.')
        if arm=='D3':
            w=model.pooling_weights(tinyfeatures).detach().float()
            proofs[arm]['weights_max_deviation_from_uniform']=float((w-1/128).abs().max())
            if proofs[arm]['weights_max_deviation_from_uniform']<=0:raise ValueError('Attention did not update.')
        del model
    write(out/'gpu_smoke.json',proofs)
    allrows={arm:[] for arm in ['D1','D3']}
    for fold in [0,1]:
        for arm in ['D1','D3']:
            folder=out/f'fold_{fold}'/arm
            result=fit_cached_head(frame,features,fold,arm,folder,origin,head_factory=make_head,loss_function=head_loss)
            if arm=='D1':
                reference=json.loads((source/f'dinov3/fold_{fold}/D1/metrics.json').read_text())
                error=abs(result['global_weighted_r2_raw']-reference['global_weighted_r2_raw'])
                write(folder/'control_reproduction.json',{'score_error':error,'tolerance':1e-5,'passes':error<1e-5})
                if error>=1e-5:raise ValueError('Uniform control did not reproduce completed study; stop comparison.')
            allrows[arm].append(pd.read_csv(folder/'val_predictions.csv'))
    frames={arm:pd.concat(rows).sort_values('image_path').reset_index(drop=True) for arm,rows in allrows.items()}
    one,two=frames['D1'],frames['D3'];keys=['image_path','Sampling_Date','fold']+TARGETS
    if not one[keys].equals(two[keys]):raise ValueError('Paired label/image/fold alignment failed.')
    paired=one[keys].copy(); scores={}
    for arm,table in frames.items():
        pred=table[['pred_'+t for t in TARGETS]].to_numpy(np.float64); truth=table[TARGETS].to_numpy(np.float64)
        scores[arm]={'global_weighted_r2':global_r2(truth,pred),'per_target':per_target(truth,pred)}
        for t in TARGETS:paired[arm+'_pred_'+t]=table['pred_'+t]
    paired.to_csv(out/'paired_oof.csv',index=False,float_format='%.17g')
    # Independent root arithmetic/group bootstrap follows collection.
    write(out/'metrics.json',{'arms':scores,'difference_D3_minus_D1':scores['D3']['global_weighted_r2']-scores['D1']['global_weighted_r2'],'images':105,'date_groups':9,'reused_development':True,'not_full_OOF':True,'competition_score':None})
    write(out/'VERIFIED_FILES_SHA256.json',{str(p.relative_to(out)):file_sha256(p) for p in out.rglob('*') if p.is_file() and p.name not in ['STATUS.json','VERIFIED_FILES_SHA256.json']})
    write(out/'STATUS.json',{'status':'completed_pending_independent_review','images':105,'folds':[0,1],'competition_score':None})
    print(json.dumps(scores),flush=True)


if __name__=='__main__': main()
