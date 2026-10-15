"""New fold/arm-specific frozen-cache head training, fixed final SWA endpoint."""
import hashlib,json,math,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.optim.swa_utils import AveragedModel
from tqdm.auto import tqdm
from . import train as common
from .data import TARGETS,WEIGHTS,file_sha256
from .official_heads import GatedRegressionHead
from .predictor_specialize import state_digest
from .final_metrics import global_r2,per_target
from .dinov3_dense import make_head,head_loss


def training_statistics(y):
 w=y.new_tensor(WEIGHTS);mean=(y*w).sum()/len(y)
 return (((y-mean).square()*w).sum()/len(y)).sqrt().clamp_min(1.),y[:,:3].median(0).values

def fit_cached_head(frame,features,fold,arm,output,origin,epochs=200,swa_last=40,batch_size=32,stop_after_updates=None,head_factory=None,loss_function=None):
 output=Path(output);output.mkdir(parents=True,exist_ok=True);device=features.device
 table_sha=hashlib.sha256(frame[['image_path','Sampling_Date','fold']+TARGETS].to_csv(index=False,float_format='%.17g').encode()).hexdigest()
 feature_sha=hashlib.sha256(features.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
 from .official_heads import feature_code_revision
 signature={'table_sha256':table_sha,'features_sha256':feature_sha,'code_sha256':feature_code_revision(),'origin':origin,'fold':int(fold),'arm':arm,'epochs':epochs,'swa_last':swa_last,'batch_size':batch_size,'lr':3e-4,'weight_decay':.01,'seed':42+int(fold),'selection':'fixed_final_swa','images':frame.image_path.tolist()}
 latest=output/'head_latest.pt'
 common.seed_all(42+int(fold));isj=arm.startswith('J_')
 training=torch.tensor(np.flatnonzero(frame.fold.to_numpy()!=fold),device=device);validation=torch.tensor(np.flatnonzero(frame.fold.to_numpy()==fold),device=device)
 y=torch.tensor(frame[TARGETS].to_numpy(),dtype=torch.float32,device=device);weights=y.new_tensor(WEIGHTS)
 scale,median=training_statistics(y[training]);model=(GatedRegressionHead(features.shape[-1],scale,median) if isj else (head_factory or make_head)(arm,scale)).to(device)
 initial=state_digest(model)
 if device.type=='cuda':
  from .final_precision import configure
  if 'rtx pro 6000' not in torch.cuda.get_device_name(device).lower() or torch.cuda.get_device_capability(device)[0]<10:raise RuntimeError('Actual RTXPRO6000 required.')
  precision=configure(model,device)
 else:precision={'active':'CPU functional test only'}
 signature['precision']=precision
 if (output/'metrics.json').exists():
  result=json.loads((output/'metrics.json').read_text())
  if result['signature']!=signature or file_sha256(output/'head_selected.pt')!=result['head_sha256']:raise ValueError('Existing completed head differs from current signature or checksum.')
  return result
 optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01);swa=AveragedModel(model)
 epoch=cursor=steps=0;order=None;history=[];epochloss=0.;epochseen=0;started=time.monotonic()
 if latest.exists():
  ck=torch.load(latest,weights_only=True,map_location=device)
  if ck['signature']!=signature:raise ValueError('Head resume signature mismatch.')
  model.load_state_dict(ck['model'],strict=True);swa.load_state_dict(ck['swa'],strict=True);optimizer.load_state_dict(ck['optimizer']);common.restore_rng(ck['rng'])
  epoch,cursor,steps,order,history,epochloss,epochseen=[ck[k] for k in ['epoch','cursor','steps','order','history','epochloss','epochseen']]
 def commit():
  common.atomic_save({'signature':signature,'model':model.state_dict(),'swa':swa.state_dict(),'optimizer':optimizer.state_dict(),'rng':common.rng_state(),'epoch':epoch,'cursor':cursor,'steps':steps,'order':order,'history':history,'epochloss':epochloss,'epochseen':epochseen},latest)
 bar=tqdm(total=epochs,initial=epoch,desc=f'fold{fold} {arm} head epochs');lastsave=time.monotonic()
 try:
  while epoch<epochs:
   if order is None:order=training[torch.randperm(len(training),device=device)].tolist();cursor=0;epochloss=0.;epochseen=0
   model.train();optimizer.param_groups[0]['lr']=3e-4*.5*(1+math.cos(math.pi*epoch/epochs))
   while cursor<len(order):
    idx=torch.tensor(order[cursor:cursor+batch_size],device=device);optimizer.zero_grad(set_to_none=True)
    with common.amp(device):
     if isj:
      rotation=torch.randint(4,(len(idx),),device=device);pred=model(features[rotation,idx]);loss=((pred.float()-y[idx])**2*weights).sum(1).mean()/scale.square()
     else:loss,_=(loss_function or head_loss)(model,features[idx],y[idx],arm)
    if not torch.isfinite(loss):raise ValueError('Nonfinite head loss.')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();steps+=1;cursor+=len(idx);epochseen+=len(idx);epochloss+=float(loss.detach())*len(idx)
    if time.monotonic()-lastsave>=60:commit();lastsave=time.monotonic()
    if stop_after_updates is not None and steps>=stop_after_updates:commit();return {'complete':False,'steps':steps,'elapsed_seconds':time.monotonic()-started}
   if epoch>=epochs-swa_last:swa.update_parameters(model)
   history.append({'epoch':epoch+1,'loss':epochloss/epochseen,'steps':steps,'elapsed_seconds':time.monotonic()-started});epoch+=1;order=None;cursor=0
   commit();pd.DataFrame(history).to_csv(output/'head_training.csv',index=False);bar.update(1);bar.set_postfix(loss=f'{history[-1]["loss"]:.5f}')
 finally:bar.close()
 if int(swa.n_averaged)!=swa_last:raise ValueError('Fixed final SWA count wrong.')
 selected=swa.module.eval()
 with torch.no_grad(),common.amp(device):
  pred=selected(features[0,validation] if isj else features[validation]).float().cpu().numpy()
  coarse=None
  if arm=='D2':_,p4=selected.predictions_at_scales(features[validation]);coarse=p4.float().cpu().numpy()
 indices=validation.cpu().numpy();rows=frame.iloc[indices][['image_path','Sampling_Date','fold']+TARGETS].copy()
 for j,t in enumerate(TARGETS):rows['pred_'+t]=pred[:,j].astype(np.float64)
 if coarse is not None:
  for j,t in enumerate(TARGETS):rows['coarse_'+t]=coarse[:,j].astype(np.float64)
 rows.to_csv(output/'val_predictions.csv',index=False,float_format='%.17g');serialized=pd.read_csv(output/'val_predictions.csv');truth=serialized[TARGETS].to_numpy(np.float64);prediction=serialized[['pred_'+t for t in TARGETS]].to_numpy(np.float64)
 score=global_r2(truth,prediction)
 from .metrics import scores
 parity=abs(score-scores(truth,prediction)['global_weighted_r2_raw'])
 if parity>1e-10:raise ValueError('Independent metric parity failed.')
 common.atomic_save({'model':selected.state_dict(),'signature':signature,'dim':features.shape[-1],'scale':scale.cpu(),'median':median.cpu(),'completed_epochs':epoch,'swa_count':int(swa.n_averaged),'precision':precision,'initial_state_sha256':initial},output/'head_selected.pt')
 result={'signature':signature,'complete':True,'completed_epochs':epoch,'swa_count':int(swa.n_averaged),'optimizer_steps':steps,'global_weighted_r2_raw':score,'per_target':per_target(truth,prediction),'metric_parity_error':parity,'training_images':len(training),'validation_images':len(validation),'parameters':sum(p.numel() for p in model.parameters()),'scale_training_only':float(scale),'elapsed_seconds':time.monotonic()-started,'precision':precision,'head_sha256':file_sha256(output/'head_selected.pt'),'initial_state_sha256':initial}
 common.write_json(output/'metrics.json',result);return result


def cache_original_features(frame,model,output,signature,branch,batch_size=2):
 """Atomic cursor with strict original-source and implementation signatures."""
 from .within_image_jigsaw import native_views,preprocess
 from .dinov3_dense import extract_dense
 output=Path(output);output.mkdir(parents=True,exist_ok=True);device=next(model.parameters()).device
 orientations=4 if branch=='J' else 1;size=224 if branch=='J' else 512
 signature={**signature,'branch':branch,'images':frame[['image_path','image_sha256']].to_dict('records'),'interpolation':'PIL_BICUBIC_native_before_resize','views':'two_complete1000_square','orientations':orientations,'normalization':'ImageNet_RGB','batch_size':batch_size}
 model.requires_grad_(False).eval()
 if device.type=='cuda':
  from .final_precision import configure
  precision=configure(model,device)
 else:precision={'active':'CPU functional test only'}
 signature['precision']=precision;path=output/'features.pt';cursor=0
 shape=(orientations,len(frame),2,model.encoder.dim) if branch=='J' else (1,len(frame),2,8,8,768)
 features=torch.zeros(shape,dtype=torch.float32)
 if path.exists():
  ck=torch.load(path,weights_only=True)
  if ck['signature']!=signature:raise ValueError('Feature-cache strict signature mismatch.')
  features=ck['features'];cursor=ck['cursor']
  if tuple(features.shape)!=shape:raise ValueError('Feature cache shape differs.')
 for start in tqdm(range(cursor,len(frame),batch_size),desc=f'{branch} original feature cache'):
  rows=frame.iloc[start:start+batch_size];tiles=torch.stack([torch.stack([preprocess(t,size) for t in native_views(p)]) for p in rows.absolute_path]).to(device)
  for rotation in range(orientations):
   rotated=torch.rot90(tiles,rotation,(-2,-1))
   with torch.no_grad(),common.amp(device):out=model(rotated) if branch=='J' else extract_dense(model,rotated)
   if not torch.isfinite(out).all():raise ValueError('Nonfinite cache features.')
   features[rotation,start:start+len(rows)]=out.float().cpu()
  cursor=start+len(rows);common.atomic_save({'signature':signature,'cursor':cursor,'features':features},path)
 manifest={'status':'completed','signature':signature,'features_sha256':file_sha256(path),'shape':list(features.shape),'cursor':cursor}
 common.write_json(output/'features_manifest.json',manifest)
 return features if branch=='J' else features[0],manifest
