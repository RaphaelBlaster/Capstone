"""One full F1/F2 dispatch after exact-package independent GPU/outer-source PASS."""
import fcntl,hashlib,json,shutil,time
from pathlib import Path
from datetime import datetime,timezone
from tools.watch_kaggle import bounded_api
from tools.run_dino_weighting import quota
from tools.run_block_shuffle_experiment import get_saved
from tools.launch_interface_inference import get_inference_source
O=Path('outputs/dino_resume_rca/full_launch');K='meghmayur1186/img2biomass-dino-finetune-f1-f2';ACTIVE={'QUEUED','RUNNING','PENDING','INITIALIZING'}
def write(n,r):
 p=O/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2));t.replace(p)
def dispatch():
 with (O/'dispatch.lock').open('a') as handle:
  fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (O/'launch.json').exists():raise ValueError('Existing dispatch receipt; never repeat write.')
  m=json.loads((O/'build_manifest.json').read_text());review=json.loads((O/'FULL_LAUNCH_VERDICT.json').read_text());gpu=json.loads((O/'GPU_RESUME_VERDICT.json').read_text());auth=json.loads((O/'USER_EXECUTION_AUTHORIZATION.json').read_text());meta=json.loads((O/'kernel-metadata.json').read_text());nb=O/meta['code_file'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
  if review.get('status')!='PASS' or review.get('notebook_sha256')!=sha(nb) or review.get('package_sha256')!=sha(O/'package.zip'):raise ValueError('Independent exact full artifact PASS required.')
  if gpu.get('verdict')!='PASS_GPU_RESUME' or gpu.get('status')!='PASS' or gpu.get('package_sha256')!=m['package_sha256'] or gpu.get('gpu_fp8_verified') is not True:raise ValueError('Actual independent resume PASS required.')
  if not auth['authorized'] or auth['mode']!='full' or auth['package_sha256']!=m['package_sha256'] or m['F3_enabled']:raise ValueError('Explicit user full authorization/F3 scope differs.')
  baseline=json.loads(Path('outputs/dino_resume_rca/prelaunch/job_scan.json').read_text());now=datetime.now(timezone.utc);since=datetime.fromisoformat(baseline['utc'])
  if not baseline['complete'] or (now-since).total_seconds()>1800:raise ValueError('Recent complete paced baseline required.')
  api=bounded_api();q=quota(api)
  if api.config_values.get('username')!='meghmayur1186' or q['reserved_seconds'] or q['remaining_seconds']<max(10800,m['estimated_budget_with_margin_seconds']):raise ValueError('Current account/quota fails.')
  # Refresh every owned reference's last-run metadata. Recheck every new or restarted
  # notebook since the complete paced baseline, plus authoritative dependencies.
  rows=[]
  for page in range(1,11):
   part=api.kernels_list(mine=True,user='meghmayur1186',page=page,page_size=100)
   if part is None:raise ValueError('Fresh owned inventory unavailable.')
   rows+=part
   if len(part)<100:break
  else:raise ValueError('Unbounded owned inventory.')
  refs={r.ref.strip('/').removeprefix('code/'):r for r in rows if r and r.ref};changed=set(refs)-set(baseline['states'])
  inventory={}
  for ref,row in refs.items():
   last=row.last_run_time
   if last is not None:
    last=last.replace(tzinfo=timezone.utc) if last.tzinfo is None else last
    if last>=since:changed.add(ref)
   inventory[ref]=str(last)
  changed.update(['meghmayur1186/img2biomass-dino-resume-rca','meghmayur1186/img2biomass-dino-anchored-finetune','meghmayur1186/img2biomass-final-dual-jepa-dino','meghmayur1186/img2biomass-dino-local-weighting'])
  states={}
  for ref in sorted(changed):
   time.sleep(4);state=api.kernels_status(ref).status.name;states[ref]=state
   if state in ACTIVE:raise ValueError('Current owned job active; no duplicate/overlap.')
  if states['meghmayur1186/img2biomass-dino-resume-rca']!='COMPLETE' or get_saved(api,'meghmayur1186/img2biomass-dino-resume-rca')['metadata']['currentVersionNumber']!=3:raise ValueError('GPU evidence version/state changed.')
  if states['meghmayur1186/img2biomass-final-dual-jepa-dino']!='COMPLETE' or get_saved(api,'meghmayur1186/img2biomass-final-dual-jepa-dino')['metadata']['currentVersionNumber']!=2:raise ValueError('D2 source version/state changed.')
  if get_inference_source(api,K) is not None:raise ValueError('Full kernel already exists; reconcile rather than recreate.')
  q=quota(api)
  if q['reserved_seconds'] or q['remaining_seconds']<10800:raise ValueError('Final fresh quota fails.')
  write('CURRENT_PREFLIGHT.json',{'utc':datetime.now(timezone.utc).isoformat(),'complete_paced_baseline':baseline['utc'],'owned_refs':len(refs),'last_run_inventory':inventory,'new_or_recent_and_dependency_states':states,'quota':q,'strategy':'fresh complete owned last-run inventory plus all changed/new refs and pinned dependency states; zero GPU reservation'})
  receipt={'kernel':K,'expected_version':1,'notebook_sha256':sha(nb),'package_sha256':sha(O/'package.zip'),'independent_resume_verdict':'PASS_GPU_RESUME','quota':q,'checked_utc':datetime.now(timezone.utc).isoformat(),'status':'dispatch_started','automatic_retries':False,'F3_enabled':False,'mode':'full','execution_cap_seconds':10800}
  shutil.copy2(nb,O/'submitted.ipynb');write('launch.json',receipt)
  try:
   ack=api.kernels_push(str(O),timeout=10800,acc='NvidiaRtxPro6000');receipt['ack']={'ref':ack.ref,'version':ack.version_number,'id':ack.kernel_id,'error_present':bool(ack.error)}
  except Exception as e:receipt['ambiguous_error_class']=type(e).__name__
  write('launch.json',receipt);actual=get_saved(api,K);r=actual['metadata'];remote=json.loads(actual['blob']['source']);local=json.loads(nb.read_text());cells=lambda n:[(c['cell_type'],''.join(c['source'])) for c in n['cells']]
  if r['currentVersionNumber']!=1 or cells(remote)!=cells(local) or r['machineShape']!='NvidiaRtxPro6000' or not r['isPrivate'] or r['enableInternet'] or not r['enableGpu'] or set(r['datasetDataSources'])!=set(meta['dataset_sources']) or r['competitionDataSources']!=meta['competition_sources']:raise ValueError('Accepted source differs; no automatic retry.')
  write('actual_saved_source.json',actual);receipt.update(status='submitted_exact_readback_verified',kernel_id=r['id']);write('launch.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':dispatch()
