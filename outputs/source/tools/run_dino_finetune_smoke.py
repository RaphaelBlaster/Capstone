"""Read-only paced preflight followed by ONE independently reviewed GPU write."""
import argparse,json,time,shutil,fcntl
from pathlib import Path
from datetime import datetime,timezone
from tools.watch_kaggle import bounded_api
from tools.run_block_shuffle_experiment import get_saved
from biomass_jepa.data import file_sha256
KERNEL='meghmayur1186/img2biomass-dino-anchored-finetune'
SOURCE='meghmayur1186/img2biomass-final-dual-jepa-dino/2'

ROOT=Path('outputs/dino_finetune/smoke_launch')
ACTIVE={'QUEUED','RUNNING','PENDING','INITIALIZING'}


def write(name,row):
    ROOT.mkdir(parents=True,exist_ok=True);p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(row,indent=2));t.replace(p)


def quota(api):
    q=api.quota_view().gpu_quota
    return {'remaining_seconds':(q.total_time_allowed-q.time_used-q.time_reserved).total_seconds(),'reserved_seconds':q.time_reserved.total_seconds()}


def scan():
    api=bounded_api()
    if api.config_values.get('username')!='meghmayur1186':raise ValueError('First account required.')
    refs=set()
    for page in range(1,11):
        rows=api.kernels_list(mine=True,user='meghmayur1186',page=page,page_size=100)
        if rows is None:raise ValueError('Cannot inventory jobs.')
        refs.update(r.ref.strip('/').removeprefix('code/') for r in rows if r and r.ref)
        if len(rows)<100:break
    else:raise ValueError('Unbounded inventory.')
    refs.update(['meghmayur1186/capstone',SOURCE.rsplit('/',1)[0]])
    states={}
    for ref in sorted(refs):
        time.sleep(4)
        try:state=api.kernels_status(ref).status.name
        except Exception as e:
            if getattr(getattr(e,'response',None),'status_code',None)==404:state='NO_ACCESSIBLE_SAVED_SESSION_HTTP404'
            else:raise RuntimeError('Job check failed: '+type(e).__name__) from None
        states[ref]=state;write('job_scan.json',{'states':states,'count':len(states),'total':len(refs),'utc':datetime.now(timezone.utc).isoformat(),'complete':False})
        if state in ACTIVE:raise ValueError('Active owned job; no overlap permitted.')
    if states[SOURCE.rsplit('/',1)[0]]!='COMPLETE':raise ValueError('Source must remain complete.')
    q=quota(api)
    if q['reserved_seconds'] or q['remaining_seconds']<3600:raise ValueError('Insufficient unreserved quota.')
    write('job_scan.json',{'states':states,'count':len(states),'total':len(refs),'utc':datetime.now(timezone.utc).isoformat(),'complete':True,'quota':q})


def dispatch():
    with (ROOT/'dispatch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'launch.json').exists():raise ValueError('Existing write receipt: do not resend.')
        m=json.loads((ROOT/'build_manifest.json').read_text());meta=json.loads((ROOT/'kernel-metadata.json').read_text());nb=ROOT/meta['code_file']
        review=json.loads((ROOT/'P0_STATIC_VERDICT.json').read_text())
        auth=json.loads((ROOT/'USER_SMOKE_AUTHORIZATION.json').read_text())
        if review['status']!='PASS' or review['package_sha256']!=m['package_sha256'] or review['notebook_sha256']!=m['reviewed_prepare_notebook_sha256']:raise ValueError('Exact package review required.')
        if file_sha256(nb)!=m['notebook_sha256'] or file_sha256(ROOT/'gpu_smoke_harness.py')!=m['diagnostic_wrapper_sha256']:raise ValueError('Smoke source changed.')
        if not auth['authorized'] or auth['mode']!='smoke' or auth['package_sha256']!=m['package_sha256'] or auth['full_training_authorized'] or auth['F3_authorized']:raise ValueError('Smoke-only authorization required.')
        if m['mode']!='smoke_only' or m['max_optimizer_updates']!=10 or m['full_training_enabled'] or m['F3_enabled']:raise ValueError('Bounded smoke required.')
        checked=json.loads((ROOT/'job_scan.json').read_text())
        if not checked['complete'] or (datetime.now(timezone.utc)-datetime.fromisoformat(checked['utc'])).total_seconds()>600:raise ValueError('Fresh complete job scan required.')
        api=bounded_api()
        q=quota(api)
        if q['reserved_seconds'] or q['remaining_seconds']<3600:raise ValueError('Current quota fails.')
        saved=get_saved(api,SOURCE.rsplit('/',1)[0])
        if saved['metadata']['currentVersionNumber']!=2 or api.kernels_status(SOURCE.rsplit('/',1)[0]).status.name!='COMPLETE':raise ValueError('Pinned source2 changed.')
        # Refuse a pre-existing new notebook instead of replacing a saved version.
        from tools.launch_interface_inference import get_inference_source
        if get_inference_source(api,KERNEL) is not None:raise ValueError('New kernel already exists; reconcile, never overwrite.')
        receipt={'kernel':KERNEL,'expected_version':1,'notebook_sha256':file_sha256(nb),'package_sha256':m['package_sha256'],'quota':q,'checked_utc':datetime.now(timezone.utc).isoformat(),'status':'dispatch_started','automatic_retries':False,'full_training_launched':False,'F3_enabled':False,'max_optimizer_updates':10}
        shutil.copy2(nb,ROOT/'submitted.ipynb');write('launch.json',receipt)
        try:
            ack=api.kernels_push(str(ROOT),timeout=1800,acc='NvidiaRtxPro6000');receipt['ack']={'ref':ack.ref,'version':ack.version_number,'id':ack.kernel_id,'error_present':bool(ack.error)}
        except Exception as e:receipt['ambiguous_error_class']=type(e).__name__
        write('launch.json',receipt)
        actual=get_saved(api,KERNEL);r=actual['metadata'];remote=json.loads(actual['blob']['source']);local=json.loads(nb.read_text())
        cells=lambda n:[(c['cell_type'],''.join(c['source'])) for c in n['cells']]
        if r['currentVersionNumber']!=1 or cells(remote)!=cells(local) or r['machineShape']!='NvidiaRtxPro6000' or not r['isPrivate'] or r['enableInternet'] or not r['enableGpu'] or set(r['datasetDataSources'])!=set(meta['dataset_sources']) or r['competitionDataSources']!=meta['competition_sources']:raise ValueError('Readback differs; never repeat push.')
        write('actual_saved_source.json',actual);receipt.update(status='submitted_exact_readback_verified',kernel_id=r['id']);write('launch.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scan',action='store_true');p.add_argument('--dispatch',action='store_true');a=p.parse_args()
    if a.scan:scan()
    if a.dispatch:dispatch()
