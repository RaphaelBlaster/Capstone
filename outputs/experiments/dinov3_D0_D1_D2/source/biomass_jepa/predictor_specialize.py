"""Predictor-only fold-filtered JEPA specialization with materialized paired masks."""
import hashlib,json,math,time
from pathlib import Path
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm
from . import train as common
from .official_ijepa import OfficialEncoder,make_predictor,OFFICIAL_REVISION,validate_ssl_masks
from .official_train import StrictOfficialCollator
from .ijepa_reference.masks import apply_masks
from .ijepa_reference.tensors import repeat_interleave_batch
from .within_image_jigsaw import permutation_bank,native_views,apply_permutation,preprocess

SOURCE_SHA='6d2b5d9c83423155b4a948822e846caaa9a6bc6be20e3fab13ca06b8e705f43d'

def state_digest(module):
 h=hashlib.sha256()
 for k,v in sorted(module.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
 return h.hexdigest()

def enable_predictor_training(predictor):
 predictor.requires_grad_(True)
 # Official fixed sinusoidal positions remain fixed; train mask token and predictor layers.
 for name,p in predictor.named_parameters():
  if name in ('predictor_pos_embed','pos_embed'):p.requires_grad_(False)
 return predictor

def load_stage1(path,device='cpu'):
 from .data import file_sha256
 if file_sha256(path)!=SOURCE_SHA:raise ValueError('Adapted checkpoint SHA differs from locked source.')
 ck=torch.load(path,weights_only=True,map_location='cpu',mmap=True)
 if ck.get('step')!=2400 or ck.get('domain_adaptation_status')!='completed_steps' or ck.get('official_repository_revision')!=OFFICIAL_REVISION or not all(k in ck for k in ['encoder','predictor','target_encoder']):raise ValueError('Missing genuine complete student/predictor/EMA Stage1.')
 enc=OfficialEncoder(224,0);teacher=OfficialEncoder(224,0);pred=make_predictor(enc)
 for model,key in [(enc,'encoder'),(pred,'predictor'),(teacher,'target_encoder')]:model.load_state_dict(ck[key],strict=True,assign=True);model.to(device)
 enc.requires_grad_(False).eval();teacher.requires_grad_(False).eval();enable_predictor_training(pred)
 return enc,pred,teacher

def exposure_schedule(frame,fold,seed=42):
 if frame.groupby('Sampling_Date')['fold'].nunique().max()!=1:raise ValueError('Date leakage across folds.')
 slots=[]
 for _,r in frame[frame.fold!=fold].sort_values('image_path').iterrows():
  for v in (0,1):
   bank=permutation_bank(r.image_sha256,v,seed)
   for slot in range(20):slots.append({'image_path':r.image_path,'image_sha256':r.image_sha256,'Sampling_Date':str(r.Sampling_Date),'parent_fold':int(r.fold),'view':v,'slot':slot,'permutation':None if slot<10 else bank[slot-10]})
 g=torch.Generator().manual_seed(seed+fold);order=torch.randperm(len(slots),generator=g).tolist()
 return [slots[i] for i in order]

def audit_schedule(frame,fold,slots):
 val=frame[frame.fold==fold];ids=set(val.image_path);dates=set(map(str,val.Sampling_Date))
 if any(s['image_path'] in ids or str(s['Sampling_Date']) in dates or s['parent_fold']==fold for s in slots):raise ValueError('Held-out parent/variant/date in specialization.')
 expected=set(frame[frame.fold!=fold].image_path)
 if {s['image_path'] for s in slots}!=expected:raise ValueError('Missing training parent.')
 for p in expected:
  for view in (0,1):
   rows=[s for s in slots if s['image_path']==p and s['view']==view]
   if sorted(s['slot'] for s in rows)!=list(range(20)):raise ValueError('Missing or duplicate exposure slot.')
   if len({tuple(s['permutation']) for s in rows if s['permutation'] is not None})!=10:raise ValueError('Ten distinct permutations required.')
 return {'validation_parent_intersection':0,'validation_date_intersection':0,'parents':len(expected),'slots':len(slots),'optimizer_steps':math.ceil(len(slots)/32)}

def materialize_schedule(frame,fold,path,seed=42,batch_size=8):
 path=Path(path);signature={'fold':fold,'seed':seed,'batch_size':batch_size,'parents':frame[['image_path','image_sha256','fold','Sampling_Date']].astype(str).to_dict('records')}
 if path.exists():
  saved=torch.load(path,weights_only=True)
  if saved['signature']!=signature:raise ValueError('Schedule signature mismatch.')
  return saved
 slots=exposure_schedule(frame,fold,seed);audit=audit_schedule(frame,fold,slots);batches=[]
 # Actual arrays stored once, replayed in both arms. Preserve caller RNG.
 rng=common.rng_state()
 try:
  for i in tqdm(range(0,len(slots),batch_size),desc=f'fold{fold} materialize paired masks'):
   digest=hashlib.sha256(json.dumps(slots[i:i+batch_size],sort_keys=True).encode()).digest();maskseed=int.from_bytes(digest[:4],'little')
   torch.manual_seed(maskseed);collator=StrictOfficialCollator();collator.reference._itr_counter.value=maskseed%1000000
   _,contexts,targets=collator([torch.zeros(3,224,224) for _ in slots[i:i+batch_size]])
   batches.append({'slots':slots[i:i+batch_size],'contexts':contexts,'targets':targets})
 finally:common.restore_rng(rng)
 saved={'signature':signature,'audit':audit,'batches':batches};common.atomic_save(saved,path);return saved

def slot_loader(image_root,arm):
 if arm not in ('control','shuffle'):raise ValueError('Unknown specialization arm.')
 def load(batch):
  images=[]
  for s in batch['slots']:
   tile=native_views(Path(image_root)/s['image_path'])[s['view']]
   if arm=='shuffle' and s['permutation'] is not None:tile=apply_permutation(tile,s['permutation'])
   images.append(preprocess(tile,224))
  return torch.stack(images)
 return load

def latent_pair(encoder,predictor,teacher,images,contexts,targets):
 validate_ssl_masks(contexts,targets,len(images),encoder.grid**2)
 with torch.no_grad():
  teacher.eval();encoder.eval();tokens=teacher(images);target=apply_masks(F.layer_norm(tokens,(tokens.shape[-1],)),targets)
  target=repeat_interleave_batch(target,len(images),repeat=len(contexts));context=encoder(images,contexts)
 predicted=predictor(context,contexts,targets)
 if target.shape!=predicted.shape:raise ValueError('Teacher/predictor mask ordering mismatch.')
 return predicted,target

def optimize_schedule(encoder,predictor,teacher,batches,loader,output,signature,effective_batch=32,stop_after=None):
 output=Path(output);output.mkdir(parents=True,exist_ok=True);device=next(predictor.parameters()).device
 encoder.requires_grad_(False).eval();teacher.requires_grad_(False).eval();enable_predictor_training(predictor).train()
 if device.type=='cuda':
  from .final_precision import configure
  if 'rtx pro 6000' not in torch.cuda.get_device_name(device).lower() or torch.cuda.get_device_capability(device)[0]<10:raise RuntimeError('Actual RTXPRO6000 Blackwell required.')
  precision=configure(predictor,device)
  precision['student_fp8']=configure(encoder,device)
 else:precision={'active':'CPU functional test only'}
 common.seed_all(42+int(signature.get('fold',0)))
 optimizer=torch.optim.AdamW([p for p in predictor.parameters() if p.requires_grad],lr=1e-5,weight_decay=.04)
 total_samples=sum(len(b['slots']) for b in batches);total_steps=math.ceil(total_samples/effective_batch)
 if {id(p) for group in optimizer.param_groups for p in group['params']}!={id(p) for p in predictor.parameters() if p.requires_grad}:raise ValueError('Optimizer must contain predictor only.')
 before_encoder=state_digest(encoder);before_teacher=state_digest(teacher);initial={k:v.detach().cpu().clone() for k,v in predictor.state_dict().items()}
 cursor=steps=0;maxgrad=0.;started=time.monotonic();latest=output/'latest.pt';logrows=[]
 if latest.exists():
  ck=torch.load(latest,weights_only=True,map_location=device)
  if ck['signature']!=signature:raise ValueError('Specialization resume signature mismatch.')
  predictor.load_state_dict(ck['predictor'],strict=True);optimizer.load_state_dict(ck['optimizer']);common.restore_rng(ck['rng']);cursor=ck['cursor'];steps=ck['steps'];logrows=ck['logrows']
  if ck['encoder_sha256']!=before_encoder or ck['teacher_sha256']!=before_teacher:raise ValueError('Resume encoders changed.')
 def save(status):
  common.atomic_save({'predictor':predictor.state_dict(),'optimizer':optimizer.state_dict(),'rng':common.rng_state(),'cursor':cursor,'steps':steps,'signature':signature,'encoder_sha256':before_encoder,'teacher_sha256':before_teacher,'status':status,'logrows':logrows},latest)
 progress=tqdm(total=total_steps,initial=steps,desc=f"JEPA {signature.get('fold','smoke')} {signature.get('arm','predictor')} optimizer updates")
 try:
  while cursor<len(batches):
   update_started=time.monotonic()
   end=cursor;count=0
   while end<len(batches) and count<effective_batch:count+=len(batches[end]['slots']);end+=1
   if count>effective_batch:raise ValueError('Microbatch boundaries do not divide effective batch.')
   multiplier=(steps+1)/20 if steps<20 else .5*(1+math.cos(math.pi*(steps-20)/max(1,total_steps-20)))
   for g in optimizer.param_groups:g['lr']=1e-5*multiplier
   optimizer.zero_grad(set_to_none=True);lossvalue=0.
   for b in batches[cursor:end]:
    images=loader(b).to(device);contexts=[m.to(device) for m in b['contexts']];targets=[m.to(device) for m in b['targets']]
    with common.amp(device):pred,target=latent_pair(encoder,predictor,teacher,images,contexts,targets)
    loss=F.smooth_l1_loss(pred.float(),target.float())*len(images)/count
    if not torch.isfinite(loss):raise ValueError('Nonfinite SSL objective.')
    loss.backward();lossvalue+=float(loss.detach())
   grads=[p.grad for p in predictor.parameters() if p.grad is not None]
   if not grads or not all(torch.isfinite(g).all() for g in grads):raise ValueError('Missing/nonfinite predictor gradients.')
   maxgrad=max(maxgrad,max(float(g.abs().max()) for g in grads));torch.nn.utils.clip_grad_norm_(predictor.parameters(),1.)
   optimizer.step();cursor=end;steps+=1
   logrows.append({'step':steps,'cursor':cursor,'loss':lossvalue,'samples':count,'elapsed_seconds':time.monotonic()-started});save('running')
   logrows[-1]['optimizer_and_checkpoint_seconds']=time.monotonic()-update_started
   import pandas as pd
   pd.DataFrame(logrows).to_csv(output/'ssl_training.csv',index=False);progress.update(1);progress.set_postfix(loss=f'{lossvalue:.5f}')
   if stop_after is not None and steps>=stop_after:break
 finally:progress.close()
 after_encoder=state_digest(encoder);after_teacher=state_digest(teacher)
 if before_encoder!=after_encoder or before_teacher!=after_teacher or any(p.grad is not None for m in (encoder,teacher) for p in m.parameters()):raise ValueError('Frozen student/teacher state or gradients changed.')
 update=max(float((v.detach().cpu()-initial[k]).abs().max()) for k,v in predictor.state_dict().items())
 if steps and update<=0:raise ValueError('Predictor did not update.')
 complete=cursor==len(batches);save('completed' if complete else 'incomplete')
 if complete:common.atomic_save({'predictor':predictor.state_dict(),'signature':signature,'steps':steps,'state_sha256':state_digest(predictor),'stage1_sha256':SOURCE_SHA},output/'predictor_specialized.pt')
 return {'steps':steps,'total_steps':total_steps,'complete':complete,'max_abs_predictor_update':update,'max_abs_gradient':maxgrad,'encoder_before_sha256':before_encoder,'encoder_after_sha256':after_encoder,'teacher_before_sha256':before_teacher,'teacher_after_sha256':after_teacher,'elapsed_seconds':time.monotonic()-started,'measured_optimizer_and_checkpoint_seconds_max':max([r.get('optimizer_and_checkpoint_seconds',0) for r in logrows] or [0]),'precision':precision}
