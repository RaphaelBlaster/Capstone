"""Locked offline dual study. Smoke is separate; full dispatch requires external P0 PASS."""
import argparse,copy,hashlib,json,math,shutil,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from . import train as common
from .data import TARGETS,WEIGHTS,file_sha256,image_table
from .official_heads import feature_code_revision
from .predictor_specialize import load_stage1,materialize_schedule,audit_schedule,optimize_schedule,slot_loader,state_digest,latent_pair,SOURCE_SHA
from .semantic_predictor_features import SemanticPredictorReadout,semantic_masks
from .within_image_jigsaw import native_views,preprocess
from .final_head import fit_cached_head,cache_original_features,training_statistics
from .dinov3_dense import load_dinov3,extract_dense,make_head,nondegeneracy_proof,head_loss

CONTRACT='DUAL-JEPA-DINO-2026-10-09-r1'
FOLD_SHA='5f00abf385baf2f20ba0a0790f97d1bf0e7349524c233ae39494bad56b254f5b'
DINO_SHA='73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c'

def write(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(path)

def audit_inputs(args):
 if file_sha256(args.adapted)!=SOURCE_SHA or file_sha256(args.folds)!=FOLD_SHA or file_sha256(args.dino)!=DINO_SHA:raise ValueError('Locked input SHA differs.')
 frame=pd.read_csv(args.folds,dtype={'Sampling_Date':str,'group_id':str}).sort_values('image_path').reset_index(drop=True)
 if len(frame)!=357 or frame.image_path.nunique()!=357 or frame.image_sha256.nunique()!=357 or frame.groupby('Sampling_Date').fold.nunique().max()!=1:raise ValueError('Exactly357 unique originalphotos and intact datefolds required.')
 frame['absolute_path']=[str(Path(args.images)/p) for p in frame.image_path]
 actual=image_table(Path(args.images)/'train.csv',args.images).sort_values('image_path').reset_index(drop=True)
 if not frame.image_path.equals(actual.image_path) or not np.array_equal(frame[TARGETS].to_numpy(),actual[TARGETS].to_numpy()):raise ValueError('Saved labels differ from original CSI CSV.')
 if frame[frame.fold.isin([0,1])].shape[0]!=105 or frame[frame.fold.isin([0,1])].Sampling_Date.nunique()!=9:raise ValueError('Expected105 images9dates reuseddevelopment.')
 for _,r in frame.iterrows():
  if file_sha256(r.absolute_path)!=r.image_sha256:raise ValueError('Original image bytes differ.')
  native_views(r.absolute_path)
 return frame,actual.attrs['target_identity_audit']

def views(rows,size):
 return torch.stack([torch.stack([preprocess(t,size) for t in native_views(p)]) for p in rows.absolute_path])

def smoke(args,frame,output):
 output=Path(output);output.mkdir(parents=True,exist_ok=True);device=torch.device('cuda');started=time.monotonic();torch.set_num_threads(2)
 enc,pred,teacher=load_stage1(args.adapted,device);initial=copy.deepcopy(pred.state_dict());proof={'contract':CONTRACT,'gpu':torch.cuda.get_device_name(),'source_code_sha256':feature_code_revision(),'stage1_sha256':SOURCE_SHA,'dino_sha256':DINO_SHA,'branches':{},'not_scientific_results':True}
 subset=pd.concat([frame[frame.fold!=0].iloc[:2],frame[frame.fold==0].iloc[:2]]).reset_index(drop=True)
 schedule=materialize_schedule(subset,0,output/'schedule.pt');jtimes=[];cachetimes=[]
 for arm in ['control','shuffle']:
  pred.load_state_dict(initial,strict=True);pred.requires_grad_(True)
  metrics=optimize_schedule(enc,pred,teacher,schedule['batches'][:8],slot_loader(args.images,arm),output/arm,{'fold':0,'arm':arm,'source_sha256':SOURCE_SHA,'smoke_only':True,'schedule_sha256':file_sha256(output/'schedule.pt')},stop_after=2)
  if metrics['steps']!=2 or not metrics['complete'] or metrics['max_abs_predictor_update']<=0 or metrics['max_abs_gradient']<=0:raise ValueError('Two-step GPU predictor update proof failed.')
  jtimes.append(metrics['measured_optimizer_and_checkpoint_seconds_max']);saved=torch.load(output/arm/'predictor_specialized.pt',weights_only=True,map_location=device);pred.load_state_dict(saved['predictor'],strict=True)
  readout=SemanticPredictorReadout(enc,pred).to(device);torch.cuda.synchronize();t=time.monotonic()
  features,manifest=cache_original_features(subset,readout,output/arm/'features',{'fold':0,'arm':arm,'predictor_sha256':file_sha256(output/arm/'predictor_specialized.pt'),'code_sha256':feature_code_revision()},'J',batch_size=2)
  torch.cuda.synchronize();cachetimes.append((time.monotonic()-t)/len(subset))
  # After-state must change semantic features versus initial weights on same unshuffled input.
  probe=views(subset.iloc[:1],224).to(device)
  with torch.no_grad(),common.amp(device):after=readout(probe).float()
  afterstate=copy.deepcopy(pred.state_dict());pred.load_state_dict(initial,strict=True)
  with torch.no_grad(),common.amp(device):before=readout(probe).float()
  pred.load_state_dict(afterstate,strict=True)
  delta=float((after-before).abs().max())
  if delta<=0:raise ValueError('Specialized predictor features did not change.')
  head=fit_cached_head(subset,features.to(device),0,'J_'+arm,output/arm/'head',{'feature_manifest_sha256':file_sha256(output/arm/'features/features_manifest.json')},stop_after_updates=2,batch_size=1)
  proof['branches']['J_'+arm]={**metrics,'semantic_after_vs_before_max_abs':delta,'head_smoke':head,'feature_shape':list(features.shape)}
 # Real actual-batch serialization included in J2 time. Release H14 before DINO.
 del enc,pred,teacher,readout,initial,afterstate,features,probe;torch.cuda.empty_cache()
 meta=json.loads(Path(args.dino_metadata).read_text());dino,origin=load_dinov3(args.dino_repository,args.dino,meta,device)
 torch.cuda.synchronize();t=time.monotonic();dense,manifest=cache_original_features(subset,dino,output/'dino_features',{'source_sha256':DINO_SHA,'code_sha256':feature_code_revision()},'D',batch_size=2)
 torch.cuda.synchronize();dtime=(time.monotonic()-t)/len(subset);dense=dense.to(device);scale,_=training_statistics(torch.tensor(subset.loc[subset.fold!=0,TARGETS].to_numpy(),dtype=torch.float32,device=device))
 testhead=make_head('D2',scale).to(device);nondeg=nondegeneracy_proof(testhead,torch.randn_like(dense));proof['dino_nondegeneracy']=nondeg;dtimes=[]
 for arm in ['D0','D1','D2']:
  head=fit_cached_head(subset,dense,0,arm,output/arm,{'feature_manifest_sha256':file_sha256(output/'dino_features/features_manifest.json')},batch_size=1,stop_after_updates=2);dtimes.append(head['elapsed_seconds']/2);proof['branches'][arm]={'head_smoke':head,'features_shape':list(dense.shape)}
 proof['dino_identity']=origin;proof['peak_vram_bytes']=torch.cuda.max_memory_allocated();proof['elapsed_seconds']=time.monotonic()-started;proof['status']='passed_actual_two_step_gpu_smoke';proof['timing']={'j_optimizer_update_seconds_max':max(jtimes),'j_cache_original_allfour_rotations_seconds':max(cachetimes),'dino_cache_image_seconds':dtime,'dino_head_update_seconds_max':max(dtimes)}
 # Full 1524 predictor updates,4x357 J caches,357Dcache,200xceil(trainN/32)headsteps.
 total_j_updates=2*(385+377);headupdates=200*(math.ceil(308/32)+math.ceil(301/32));estimated=total_j_updates*max(jtimes)+4*357*max(cachetimes)+357*dtime+headupdates*(2*max(dtimes)+3*max(dtimes))+1800
 gate={'estimated_seconds':estimated,'margin20_percent_seconds':estimated*1.2,'remaining_quota_snapshot_seconds':args.remaining_seconds,'session_cap_seconds':21600,'passes':estimated*1.2<min(args.remaining_seconds-1800,21600-600),'timing':proof['timing'],'disk_free_bytes':shutil.disk_usage(output).free,'gpu_peak_bytes':proof['peak_vram_bytes'],'full_training_not_launched':True}
 write(output/'gpu_smoke.json',proof);write(output/'resource_gate.json',gate)
 return proof,gate

def latent_diagnostics(enc,pred,teacher,frame,fold,output):
 pred.requires_grad_(False).eval();device=next(enc.parameters()).device;rows=[]
 for start in range(0,len(frame),2):
  chunk=frame.iloc[start:start+2];image=views(chunk,224).flatten(0,1).to(device);c,t=semantic_masks(len(image),device)
  with torch.no_grad(),common.amp(device):p,y=latent_pair(enc,pred,teacher,image,c,t)
  # target-major layout; diagnostics only, no optimizer.
  p=p.reshape(4,len(chunk),2,42,-1).float();y=y.reshape_as(p).float()
  losses=F.smooth_l1_loss(p,y,reduction='none').mean((0,2,3,4));cosine=F.cosine_similarity(p,y,dim=-1).mean((0,2,3))
  for r,l,cvalue in zip(chunk.itertuples(),losses.tolist(),cosine.tolist()):rows.append({'image_path':r.image_path,'Sampling_Date':r.Sampling_Date,'fold':fold,'latent_smooth_l1':l,'latent_cosine':cvalue})
 pd.DataFrame(rows).to_csv(output/'heldout_latent_diagnostics.csv',index=False)

def full(args,frame,output,smokeproof,gate):
 if not gate['passes']:raise RuntimeError('Both branches cannot fit current quota/session budget.')
 device=torch.device('cuda');output=Path(output);write(output/'STATUS.json',{'status':'running','contract':CONTRACT,'started_utc':time.time(),'not_full_OOF':True})
 provenance=output/'provenance';provenance.mkdir(parents=True,exist_ok=True);frame.to_csv(provenance/'folds.csv',index=False)
 write(provenance/'CONFIG_LOCK.json',{'contract':CONTRACT,'folds':[0,1],'source_code_sha256':feature_code_revision(),'predictor_steps':{'fold0':385,'fold1':377},'stage1_repeated':False,'epochs':200,'swa':40,'lambda':.10,'DINO':'officialB16_512','progress_bars':True,'checkpoint_every_predictor_update':True})
 write(output/'jepa/stage1_reused.json',{'checkpoint_sha256':SOURCE_SHA,'step':2400,'unique_ssl_images':30663,'repeated_training':False})
 for fold in [0,1]:
  schedulepath=output/'jepa'/f'fold_{fold}'/'paired_schedule.pt';schedule=materialize_schedule(frame,fold,schedulepath);audit=audit_schedule(frame,fold,[s for b in schedule['batches'] for s in b['slots']]);write(provenance/f'leakage_fold{fold}.json',audit)
  for arm in ['control','shuffle']:
   folder=output/'jepa'/f'fold_{fold}'/arm;enc,pred,teacher=load_stage1(args.adapted,device)
   initial=state_digest(pred);signature={'fold':fold,'arm':arm,'stage1_sha256':SOURCE_SHA,'initial_predictor_state_sha256':initial,'schedule_sha256':file_sha256(schedulepath),'source_code_sha256':feature_code_revision(),'effective_batch':32,'microbatch':8,'lr':1e-5,'wd':.04,'warmup_updates':20,'total_steps':audit['optimizer_steps']}
   proof=optimize_schedule(enc,pred,teacher,schedule['batches'],slot_loader(args.images,arm),folder,signature)
   if not proof['complete'] or proof['steps']!=audit['optimizer_steps']:raise RuntimeError('Incomplete paired predictor schedule.')
   write(folder/'predictor_update_proof.json',proof)
   specialized=torch.load(folder/'predictor_specialized.pt',weights_only=True,map_location=device);pred.load_state_dict(specialized['predictor'],strict=True)
   latent_diagnostics(enc,pred,teacher,frame[frame.fold==fold],fold,folder);del teacher;torch.cuda.empty_cache()
   readout=SemanticPredictorReadout(enc,pred).to(device);cacheorigin={**signature,'predictor_specialized_sha256':file_sha256(folder/'predictor_specialized.pt'),'semantic_masks':'four6x7_corners_context88','state_sha256':state_digest(pred)}
   features,manifest=cache_original_features(frame,readout,folder,cacheorigin,'J')
   del readout,enc,pred;torch.cuda.empty_cache()
   fit_cached_head(frame,features.to(device),fold,'J_'+arm,folder,{'manifest_sha256':file_sha256(folder/'features_manifest.json'),'predictor_sha256':cacheorigin['predictor_specialized_sha256']});del features;torch.cuda.empty_cache()
 dino,origin=load_dinov3(args.dino_repository,args.dino,json.loads(Path(args.dino_metadata).read_text()),device)
 write(output/'dinov3/checkpoint.json',origin)
 features,manifest=cache_original_features(frame,dino,output/'dinov3',{'checkpoint_sha256':DINO_SHA,'source_code_sha256':feature_code_revision()},'D');write(output/'dinov3/cache_manifest.json',manifest);del dino;torch.cuda.empty_cache();features=features.to(device)
 for fold in [0,1]:
  for arm in ['D0','D1','D2']:fit_cached_head(frame,features,fold,arm,output/'dinov3'/f'fold_{fold}'/arm,{'manifest_sha256':file_sha256(output/'dinov3/cache_manifest.json')})
 from .final_figures import report
 report(output,frame)
 hashes={str(p.relative_to(output)):file_sha256(p) for p in output.rglob('*') if p.is_file() and p.name not in ['VERIFIED_FILES_SHA256.json','STATUS.json']};write(output/'VERIFIED_FILES_SHA256.json',hashes)
 write(output/'STATUS.json',{'status':'completed_pending_independent_P1','contract':CONTRACT,'images':105,'date_groups':9,'folds':[0,1],'competition_score':None,'independent_P1':'pending'})

def prepare_full(output,frame):
 output=Path(output);provenance=output/'provenance';provenance.mkdir(parents=True,exist_ok=True);frame.to_csv(provenance/'folds.csv',index=False)
 write(output/'STATUS.json',{'status':'running','contract':CONTRACT,'not_full_OOF':True})
 write(output/'jepa/stage1_reused.json',{'checkpoint_sha256':SOURCE_SHA,'step':2400,'unique_ssl_images':30663,'repeated_training':False})

def run_j_arm(args,frame,output,fold,arm):
 output=Path(output);device=torch.device('cuda');provenance=output/'provenance'
 schedulepath=output/'jepa'/f'fold_{fold}'/'paired_schedule.pt';schedule=materialize_schedule(frame,fold,schedulepath);audit=audit_schedule(frame,fold,[s for b in schedule['batches'] for s in b['slots']]);write(provenance/f'leakage_fold{fold}.json',audit)
 folder=output/'jepa'/f'fold_{fold}'/arm;enc,pred,teacher=load_stage1(args.adapted,device)
 initial=state_digest(pred);signature={'fold':fold,'arm':arm,'stage1_sha256':SOURCE_SHA,'initial_predictor_state_sha256':initial,'schedule_sha256':file_sha256(schedulepath),'source_code_sha256':feature_code_revision(),'effective_batch':32,'microbatch':8,'lr':1e-5,'wd':.04,'warmup_updates':20,'total_steps':audit['optimizer_steps']}
 proof=optimize_schedule(enc,pred,teacher,schedule['batches'],slot_loader(args.images,arm),folder,signature)
 if not proof['complete'] or proof['steps']!=audit['optimizer_steps']:raise RuntimeError('Incomplete paired predictor schedule.')
 write(folder/'predictor_update_proof.json',proof)
 specialized=torch.load(folder/'predictor_specialized.pt',weights_only=True,map_location=device);pred.load_state_dict(specialized['predictor'],strict=True)
 latent_diagnostics(enc,pred,teacher,frame[frame.fold==fold],fold,folder);del teacher;torch.cuda.empty_cache()
 readout=SemanticPredictorReadout(enc,pred).to(device);cacheorigin={**signature,'predictor_specialized_sha256':file_sha256(folder/'predictor_specialized.pt'),'semantic_masks':'four6x7_corners_context88','state_sha256':state_digest(pred)}
 features,manifest=cache_original_features(frame,readout,folder,cacheorigin,'J')
 del readout,enc,pred;torch.cuda.empty_cache()
 fit_cached_head(frame,features.to(device),fold,'J_'+arm,folder,{'manifest_sha256':file_sha256(folder/'features_manifest.json'),'predictor_sha256':cacheorigin['predictor_specialized_sha256']});del features;torch.cuda.empty_cache()

def run_d_cache(args,frame,output):
 output=Path(output);device=torch.device('cuda')
 dino,origin=load_dinov3(args.dino_repository,args.dino,json.loads(Path(args.dino_metadata).read_text()),device)
 write(output/'dinov3/checkpoint.json',origin)
 features,manifest=cache_original_features(frame,dino,output/'dinov3',{'checkpoint_sha256':DINO_SHA,'source_code_sha256':feature_code_revision()},'D');write(output/'dinov3/cache_manifest.json',manifest);del dino;torch.cuda.empty_cache()
 return features,manifest

def run_d_heads(args,frame,output,features):
 output=Path(output);device=torch.device('cuda');features=features.to(device)
 for fold in [0,1]:
  for arm in ['D0','D1','D2']:fit_cached_head(frame,features,fold,arm,output/'dinov3'/f'fold_{fold}'/arm,{'manifest_sha256':file_sha256(output/'dinov3/cache_manifest.json')})

def finish(output,frame):
 output=Path(output)
 hashes={str(p.relative_to(output)):file_sha256(p) for p in output.rglob('*') if p.is_file() and p.name not in ['VERIFIED_FILES_SHA256.json','STATUS.json']};write(output/'VERIFIED_FILES_SHA256.json',hashes)
 write(output/'STATUS.json',{'status':'completed_pending_independent_P1','contract':CONTRACT,'images':105,'date_groups':9,'folds':[0,1],'competition_score':None,'independent_P1':'pending'})

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','full'],required=True)
 for n in ['adapted','folds','images','dino','dino-repository','dino-metadata','output']:p.add_argument('--'+n,required=True)
 p.add_argument('--remaining-seconds',type=float,required=True);args=p.parse_args();output=Path(args.output)
 if not torch.cuda.is_available() or 'rtx pro 6000' not in torch.cuda.get_device_name().lower() or torch.cuda.get_device_capability()[0]<10:raise RuntimeError('Actual RTXPRO6000 Blackwell required. No training fallback.')
 print('ACTUAL GPU: '+torch.cuda.get_device_name(),flush=True);frame,audit=audit_inputs(args);write(output/'provenance/target_identity_audit.json',audit)
 write(output/'provenance/environment.json',{'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'cuda':torch.version.cuda,'python':__import__('sys').version,'fp8':'eligible_linears','loss':'FP32','master_parameters':'FP32'})
 write(output/'provenance/inputs_sha256.json',{'adapted':file_sha256(args.adapted),'folds':file_sha256(args.folds),'dino':file_sha256(args.dino),'csiro_labels':file_sha256(Path(args.images)/'train.csv')})
 try:
  proof,gate=smoke(args,frame,output/'_smoke');write(output/'verifier/gpu_smoke.json',proof);write(output/'verifier/resource_gate.json',gate)
  if args.mode=='full':full(args,frame,output,proof,gate)
  else:write(output/'STATUS.json',{'status':'smoke_completed_no_full_training','P0_review':'required','resource_gate':gate})
 except Exception as e:write(output/'STATUS.json',{'status':'incomplete','error_class':type(e).__name__,'message':str(e)[:300]});raise

if __name__=='__main__':main()
