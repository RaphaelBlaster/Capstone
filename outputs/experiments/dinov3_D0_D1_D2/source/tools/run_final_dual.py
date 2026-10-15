"""Single-write offline dispatcher. Full run requires independent P0 + real GPU smoke."""
import argparse,fcntl,json,shutil,time
from pathlib import Path
from datetime import datetime,timezone
from tools.watch_kaggle import bounded_api
from tools.run_predictor_interface import remaining_quota
from tools.run_block_shuffle_experiment import get_saved
from tools.launch_interface_inference import get_inference_source
from tools.build_final_dual import KERNEL,SOURCE
from biomass_jepa.data import file_sha256
from biomass_jepa.official_heads import feature_code_revision
ACTIVE={'QUEUED','RUNNING','PENDING','INITIALIZING'}

def write(path,row):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(row,indent=2));tmp.replace(path)

def conservative_resource_gate(proof,remaining_seconds):
 import math
 timing=proof['timing']
 jhead=max(proof['branches'][arm]['head_smoke']['elapsed_seconds']/proof['branches'][arm]['head_smoke']['steps'] for arm in ('J_control','J_shuffle'))
 dhead=timing['dino_head_update_seconds_max']
 head_updates=200*(math.ceil(308/32)+math.ceil(301/32))
 estimated=2*(385+377)*timing['j_optimizer_update_seconds_max']+4*357*timing['j_cache_original_allfour_rotations_seconds']+357*timing['dino_cache_image_seconds']+head_updates*(2*jhead+3*dhead)+1800
 margin=estimated*1.2
 return {'estimated_seconds':estimated,'margin20_percent_seconds':margin,'passes':margin<min(remaining_seconds-1800,21600-600),'remaining_quota_snapshot_seconds':remaining_seconds,'session_cap_seconds':21600,'basis':'actual measured separate JEPA/DINO head costs; fixed protocol counts; setup reserve and20%margin','j_head_update_seconds_max':jhead,'dino_head_update_seconds_max':dhead}

def validate_gate(root,mode,manifest):
 if manifest['source_code_sha256']!=feature_code_revision():raise ValueError('Notebook package is stale.')
 proof=json.loads((root/'provenance/stage1_validation.json').read_text());dino=json.loads((root/'provenance/dino_cpu_validation.json').read_text())
 if not proof.get('strict_student_predictor_teacher_loaded') or not dino.get('strict_state_load',dino.get('strict_load',False)):raise ValueError('Actual checkpoint strict-load proof required.')
 if mode=='full':
  review=json.loads((root/'verifier/REPRODUCIBILITY_CHECKS.json').read_text());text=(root/'verifier/ARCHITECTURE_VERDICT.md').read_text()
  smoke=json.loads((root/'smoke_results/final_dual_study/verifier/gpu_smoke.json').read_text());gate=json.loads((root/'smoke_results/final_dual_study/verifier/resource_gate.json').read_text())
  if review.get('overall_verdict')!='PASS' or review.get('reviewed_full_notebook_sha256')!=manifest['notebook_sha256'] or 'READY_FOR_GPU' not in text:raise ValueError('Independent P0 PASS for exact full notebook required.')
  if smoke['status']!='passed_actual_two_step_gpu_smoke' or smoke['source_code_sha256']!=manifest['source_code_sha256'] or not gate['passes']:raise ValueError('Actual GPU smoke/resource gate must pass same source.')

def dispatch(base,mode):
 root=Path(base);folder=root/(mode+'_launch');folder.mkdir(exist_ok=True)
 with (folder/'dispatch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (folder/'launch.json').exists():raise ValueError('Existing accepted/ambiguous write journal: reconcile, never resend.')
  manifest=json.loads((folder/'build_manifest.json').read_text());meta=json.loads((folder/'kernel-metadata.json').read_text());nb=folder/meta['code_file']
  if manifest['mode']!=mode or file_sha256(nb)!=manifest['notebook_sha256']:raise ValueError('Prepared source bytes changed.')
  validate_gate(root,mode,manifest);api=bounded_api()
  if api.config_values.get('username')!='meghmayur1186':raise ValueError('Authorized first account required.')
  existing=get_inference_source(api,KERNEL)
  if mode=='smoke' and existing is not None:raise ValueError('New notebook already exists; no duplicate smoke.')
  if mode=='full':
   state=api.kernels_status(KERNEL).status.name
   if state!='COMPLETE':raise ValueError('Prior smoke version must finish before long run.')
   saved=get_saved(api,KERNEL)
   if saved['metadata']['currentVersionNumber']!=1:raise ValueError('Unexpected existing notebook version.')
  refs=set()
  for page in range(1,11):
   rows=api.kernels_list(mine=True,user='meghmayur1186',page=page,page_size=100)
   if rows is None:raise ValueError('Cannot inspect current owned notebooks.')
   refs.update(x.ref.strip('/').removeprefix('code/') for x in rows if x is not None and x.ref)
   if len(rows)<100:break
  else:raise ValueError('Owned notebook inventory exceeds boundedscan.')
  refs.update({'meghmayur1186/capstone',SOURCE.rsplit('/',1)[0],'meghmayur1186/img2biomass-i-jepa-frozen-encoder-head','meghmayur1186/img2biomass-jepa-block-shuffle'})
  states={}
  for ref in sorted(refs):
   time.sleep(4)
   write(folder/'job_scan.json',{'states':states,'checking':ref,'checked_count':len(states),'total_count':len(refs),'utc':datetime.now(timezone.utc).isoformat()})
   try:state=api.kernels_status(ref).status.name
   except Exception as e:
    if getattr(getattr(e,'response',None),'status_code',None)==404:states[ref]='NO_ACCESSIBLE_SAVED_SESSION_HTTP404';continue
    raise
   states[ref]=state
   if state in ACTIVE:raise ValueError('Owned saved job active; no duplicate or overlap.')
  source_saved=get_saved(api,SOURCE.rsplit('/',1)[0])
  if source_saved['metadata']['currentVersionNumber']!=2 or states[SOURCE.rsplit('/',1)[0]]!='COMPLETE':raise ValueError('Verified Stage1 source version2 must remain complete.')
  quota=remaining_quota(api);reserved=api.quota_view().gpu_quota.time_reserved.total_seconds();cap=1800 if mode=='smoke' else 21600
  if reserved!=0 or quota<cap+1800:raise ValueError('Quota/reservation not sufficient.')
  if mode=='full':
   budget=manifest.get('quota_budget_lower_bound',0)
   if budget<=0 or quota<budget:raise ValueError('Current authoritative quota is below notebook budget bound; stale source cannot authorize compute.')
   actualproof=json.loads((root/'smoke_results/final_dual_study/verifier/gpu_smoke.json').read_text())
   gate=conservative_resource_gate(actualproof,quota)
   write(folder/'conservative_resource_gate.json',gate)
   if not gate['passes']:raise ValueError('Fresh quota invalidates conservative resource gate.')
  receipt={'status':'dispatch_started','mode':mode,'kernel':KERNEL,'expected_version':1 if mode=='smoke' else 2,'notebook_sha256':file_sha256(nb),'package_sha256':manifest['embedded_package_sha256'],'source_code_sha256':manifest['source_code_sha256'],'quota_before_seconds':quota,'reserved_before_seconds':reserved,'job_states':states,'cap_seconds':cap,'checked_utc':datetime.now(timezone.utc).isoformat(),'automatic_retries':False}
  validate_gate(root,mode,manifest)
  shutil.copy2(nb,folder/'submitted.ipynb');write(folder/'launch.json',receipt)
  try:
   result=api.kernels_push(str(folder),timeout=cap,acc='NvidiaRtxPro6000');receipt['ack']={'ref':result.ref,'version':result.version_number,'id':result.kernel_id,'error_present':bool(result.error)}
  except Exception as e:receipt['ambiguous_write_error_class']=type(e).__name__
  write(folder/'launch.json',receipt)
  actual=get_saved(api,KERNEL);settings=actual['metadata'];remote=json.loads(actual['blob']['source']);local=json.loads(nb.read_text());cells=lambda n:[(c['cell_type'],''.join(c['source'])) for c in n['cells']]
  if settings['currentVersionNumber']!=receipt['expected_version'] or cells(local)!=cells(remote) or settings['machineShape']!='NvidiaRtxPro6000' or not settings['isPrivate'] or settings['enableInternet'] or not settings['enableGpu'] or set(settings['datasetDataSources'])!=set(meta['dataset_sources']) or settings['competitionDataSources']!=meta['competition_sources']:raise ValueError('Saved source/settings differ. Do not repeat write.')
  write(folder/'actual_saved_source.json',actual);receipt.update(status='submitted_exact_readback_verified',kernel_id=settings['id']);write(folder/'launch.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='outputs/final_dual_study');p.add_argument('--mode',choices=['smoke','full'],required=True);a=p.parse_args();dispatch(a.root,a.mode)
