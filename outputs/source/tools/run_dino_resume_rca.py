"""One journaled bounded diagnostic version; never dispatches full training."""
import fcntl,json,shutil,hashlib
from pathlib import Path
from datetime import datetime,timezone
from tools.watch_kaggle import bounded_api
from tools.run_dino_weighting import quota
from tools.run_block_shuffle_experiment import get_saved
from tools.launch_interface_inference import get_inference_source
KERNEL='meghmayur1186/img2biomass-dino-resume-rca'
def write(p,r):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2));t.replace(p)
def dispatch(stage):
 phase='red' if stage.startswith('red') else 'green'
 root=Path('outputs/dino_resume_rca')/stage/'launch'
 with (root/'dispatch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (root/'launch.json').exists():raise ValueError('Existing receipt; never resend.')
  scan=json.loads(Path('outputs/dino_resume_rca/prelaunch/job_scan.json').read_text())
  if not scan['complete'] or (datetime.now(timezone.utc)-datetime.fromisoformat(scan['utc'])).total_seconds()>600:raise ValueError('Fresh complete job scan required.')
  manifest=json.loads((root/'build_manifest.json').read_text());meta=json.loads((root/'kernel-metadata.json').read_text());nb=root/meta['code_file'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
  if sha(nb)!=manifest['notebook_sha256'] or manifest['max_updates']!=7 or manifest['full_training_enabled'] or manifest['F3_enabled']:raise ValueError('Bounded exact notebook required.')
  api=bounded_api();q=quota(api)
  if api.config_values.get('username')!='meghmayur1186' or q['reserved_seconds'] or q['remaining_seconds']<3600:raise ValueError('Account/quota failed.')
  for ref,version,state in [('meghmayur1186/img2biomass-final-dual-jepa-dino',2,'COMPLETE'),('meghmayur1186/img2biomass-dino-anchored-finetune',1,'ERROR')]:
   if get_saved(api,ref)['metadata']['currentVersionNumber']!=version or api.kernels_status(ref).status.name!=state:raise ValueError('Pinned prior source/version/state changed.')
  previous=get_inference_source(api,KERNEL);version={'red':1,'red_retry':2,'green':3}[stage]
  if stage=='red' and previous is not None:raise ValueError('Diagnostic already exists; reconcile.')
  if stage!='red':
   actual_previous=get_saved(api,KERNEL)['metadata']['currentVersionNumber'];actual_state=api.kernels_status(KERNEL).status.name
   if actual_previous!=version-1 or actual_state not in ['COMPLETE','ERROR']:raise ValueError('Prior diagnostic must be confirmed terminal previous version.')
  receipt={'kernel':KERNEL,'expected_version':version,'phase':phase,'notebook_sha256':sha(nb),'package_sha256':manifest['package_sha256'],'quota':q,'checked_utc':datetime.now(timezone.utc).isoformat(),'status':'dispatch_started','automatic_retries':False,'full_training_launched':False,'F3_enabled':False,'max_optimizer_updates':7}
  shutil.copy2(nb,root/'submitted.ipynb');write(root/'launch.json',receipt)
  try:
   ack=api.kernels_push(str(root),timeout=1800,acc='NvidiaRtxPro6000');receipt['ack']={'ref':ack.ref,'version':ack.version_number,'id':ack.kernel_id,'error_present':bool(ack.error)}
  except Exception as e:receipt['ambiguous_error_class']=type(e).__name__
  write(root/'launch.json',receipt);actual=get_saved(api,KERNEL);r=actual['metadata'];remote=json.loads(actual['blob']['source']);local=json.loads(nb.read_text());cells=lambda n:[(c['cell_type'],''.join(c['source'])) for c in n['cells']]
  if r['currentVersionNumber']!=version or cells(remote)!=cells(local) or r['machineShape']!='NvidiaRtxPro6000' or not r['isPrivate'] or r['enableInternet'] or not r['enableGpu'] or set(r['datasetDataSources'])!=set(meta['dataset_sources']) or r['competitionDataSources']!=meta['competition_sources']:raise ValueError('Readback differs; no retry.')
  write(root/'actual_saved_source.json',actual);receipt.update(status='submitted_exact_readback_verified',kernel_id=r['id']);write(root/'launch.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':
 import sys
 dispatch(sys.argv[1])
