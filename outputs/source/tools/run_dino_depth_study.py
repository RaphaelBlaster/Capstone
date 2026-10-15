"""Root-only single depth dispatch after exact independent review and fresh preflight."""
import fcntl
import hashlib
import json
import shutil
from datetime import datetime,timezone
from pathlib import Path
from tools.watch_kaggle import bounded_api
from tools.run_dino_weighting import quota
from tools.run_block_shuffle_experiment import get_saved
from tools.launch_interface_inference import get_inference_source
from tools.build_dino_depth_study import KERNEL

OUT=Path('outputs/dino_depth_study/launch')
def write(name,row):
    p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(row,indent=2));t.replace(p)
def dispatch():
    with (OUT/'dispatch.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'launch.json').exists():raise ValueError('Existing write receipt; never duplicate.')
        m=json.loads((OUT/'build_manifest.json').read_text());review=json.loads((OUT/'independent_review.json').read_text());meta=json.loads((OUT/'kernel-metadata.json').read_text());nb=OUT/meta['code_file']
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        if review.get('status')!='PASS' or review.get('depths')!=[2,4,8] or review.get('package_sha256')!=sha(OUT/'package.zip') or review.get('notebook_sha256')!=sha(nb):raise ValueError('Exact independent notebook/package PASS required.')
        if m['F3_enabled'] or not m['no_convolution'] or m['package_sha256']!=sha(OUT/'package.zip') or m['notebook_sha256']!=sha(nb):raise ValueError('Immutable depth manifest differs.')
        scan=json.loads((OUT/'job_scan.json').read_text())
        if not scan['complete'] or (datetime.now(timezone.utc)-datetime.fromisoformat(scan['utc'])).total_seconds()>600:raise ValueError('Fresh complete job scan required.')
        if any(s in ('QUEUED','RUNNING','PENDING','INITIALIZING') for s in scan['states'].values()):raise ValueError('Active owned job; no overlap.')
        api=bounded_api()
        if api.config_values.get('username')!='meghmayur1186':raise ValueError('First account required.')
        source='meghmayur1186/img2biomass-final-dual-jepa-dino'
        if api.kernels_status(source).status.name!='COMPLETE' or get_saved(api,source)['metadata']['currentVersionNumber']!=2:raise ValueError('Verified D2 source2 changed.')
        if get_inference_source(api,KERNEL) is not None:raise ValueError('New depth kernel already exists; reconcile, never replace.')
        q=quota(api)
        if q['reserved_seconds'] or q['remaining_seconds']<m['execution_cap_seconds']:raise ValueError('Fresh unreserved quota below depth cap.')
        receipt={'kernel':KERNEL,'expected_version':1,'notebook_sha256':sha(nb),'package_sha256':sha(OUT/'package.zip'),'quota':q,'checked_utc':datetime.now(timezone.utc).isoformat(),'status':'dispatch_started','depths':[2,4,8],'folds':[0,1],'epochs':20,'automatic_retries':False,'conditional_full_on_runtime_gates':True,'F3_enabled':False}
        shutil.copy2(nb,OUT/'submitted.ipynb');write('launch.json',receipt)
        try:
            ack=api.kernels_push(str(OUT),timeout=m['execution_cap_seconds'],acc='NvidiaRtxPro6000');receipt['ack']={'ref':ack.ref,'version':ack.version_number,'id':ack.kernel_id,'error_present':bool(ack.error)}
        except Exception as error:receipt['ambiguous_error_class']=type(error).__name__
        write('launch.json',receipt)
        actual=get_saved(api,KERNEL);r=actual['metadata'];remote=json.loads(actual['blob']['source']);local=json.loads(nb.read_text())
        cells=lambda n:[(c['cell_type'],''.join(c['source'])) for c in n['cells']]
        if r['currentVersionNumber']!=1 or cells(remote)!=cells(local) or r['machineShape']!='NvidiaRtxPro6000' or not r['isPrivate'] or r['enableInternet'] or not r['enableGpu'] or set(r['datasetDataSources'])!=set(meta['dataset_sources']) or r['competitionDataSources']!=meta['competition_sources']:raise ValueError('Accepted source/settings differ; no retry.')
        write('actual_saved_source.json',actual);receipt.update(status='submitted_exact_readback_verified',kernel_id=r['id']);write('launch.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':dispatch()
