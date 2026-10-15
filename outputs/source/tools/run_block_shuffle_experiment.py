"""One journaled offline Kaggle launch; no retries, SSL or competition submission."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import requests

from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
from biomass_jepa.data import file_sha256
from tools.build_block_shuffle_experiment import KERNEL,SOURCE,SOURCE_SHA,FOLDS_SHA,CAP
from tools.launch_interface_inference import get_inference_source
from tools.run_predictor_interface import remaining_quota
from tools.watch_kaggle import bounded_api

SOURCE_KERNEL=SOURCE.rsplit('/',1)[0]
ACTIVE={'QUEUED','PENDING','INITIALIZING','RUNNING'}


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def get_saved(api,kernel):
    req=ApiGetKernelRequest();req.user_name,req.kernel_slug=kernel.split('/')
    with api.build_kaggle_client() as client:
        return client.kernels.kernels_api_client.get_kernel(req).to_dict()


def reserved_gpu_seconds(api):
    return api.quota_view().gpu_quota.time_reserved.total_seconds()


def validate_prepared(root,folder,settings):
    expected={'id':KERNEL,'title':'Img2BioMass JEPA Block Shuffle','is_private':True,
        'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'machine_shape':'NvidiaRtxPro6000',
        'dataset_sources':['dheerajkannaujiya/csiro','meghmayur1186/img2biomass-offline-deps'],
        'competition_sources':['arc-prize-2026-arc-agi-3'],'kernel_sources':[SOURCE]}
    if any(settings.get(key)!=value for key,value in expected.items()):
        raise ValueError('Prepared immutable offline source/settings differ.')
    readiness=json.loads((folder/'verified_smoke.json').read_text())
    if (readiness['status']!='passed_functional_smoke_not_scientific_results'
            or readiness['notebook_sha256']!=file_sha256(folder/settings['code_file'])):
        raise ValueError('Fresh notebook smoke evidence required.')


def validate_local_evidence(root):
    source=root/'outputs/block_shuffle_preflight/verified_source_v2/adapted.pt'
    proof=json.loads((source.parent/'checkpoint_verification.json').read_text())
    if (proof['status']!='checkpoint_bytes_and_strict_student_encoder_verified'
            or proof['origin']['source_sha256']!=SOURCE_SHA or proof['origin']['grass_steps']!=2400
            or proof['origin']['dim']!=1280 or not proof['frozen'] or file_sha256(source)!=SOURCE_SHA):
        raise ValueError('Strict verified adapted student source required.')


def reconcile(folder,api):
    """Readback only, even after an ambiguous acknowledgement; never another write."""
    receipt=json.loads((folder/'launch.json').read_text())
    actual=get_saved(api,KERNEL);meta=actual['metadata']
    settings=json.loads((folder/'kernel-metadata.json').read_text())
    local=json.loads((folder/'submitted_block_shuffle_experiment.ipynb').read_text())
    remote=json.loads(actual['blob']['source'])
    cells=lambda nb:[(c['cell_type'],''.join(c['source'])) for c in nb['cells']]
    if (meta['ref']!=KERNEL or meta['currentVersionNumber']!=1
            or meta['machineShape']!='NvidiaRtxPro6000' or not meta['enableGpu']
            or not meta['isPrivate'] or meta['enableInternet'] or meta.get('enableTpu',False)
            or set(meta['datasetDataSources'])!=set(settings['dataset_sources'])
            or meta['competitionDataSources']!=settings['competition_sources']
            or meta['kernelDataSources']!=[SOURCE_KERNEL] or cells(local)!=cells(remote)
            or file_sha256(folder/'submitted_block_shuffle_experiment.ipynb')!=receipt['notebook_sha256']):
        raise ValueError('Accepted write readback differs; do not retry or adopt.')
    write_json(folder/'actual_saved_source_v1.json',actual)
    receipt.update(status='submitted_exact_source_readback_verified_actual_training_unverified',
        kernel_id=meta['id'],version=1,readback_utc=datetime.now(timezone.utc).isoformat(),
        input_version_note='SDK lists input slug only; runtime exact source and fold SHA verify pinned content.')
    write_json(folder/'launch.json',receipt)
    return receipt


def launch(root,api):
    root=Path(root);folder=root/'outputs/block_shuffle_launch'
    with (folder/'dispatch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'launch.json').exists():raise ValueError('Existing dispatch journal; never launch twice.')
        if api.config_values.get('username')!='meghmayur1186':raise ValueError('Existing authorized account1 required.')
        settings=json.loads((folder/'kernel-metadata.json').read_text())
        validate_prepared(root,folder,settings);validate_local_evidence(root)
        if get_inference_source(api,KERNEL) is not None:raise ValueError('Existing experiment must not be overwritten.')
        states={}
        required={SOURCE_KERNEL,'meghmayur1186/capstone','meghmayur1186/img2biomass-i-jepa-frozen-encoder-head'}
        refs=set(required)
        # Inspect all authenticated owned saved notebooks, not just stale named jobs.
        for page in range(1,11):
            rows=api.kernels_list(mine=True,user='meghmayur1186',page=page,page_size=100)
            if rows is None:raise ValueError('Cannot verify owned notebook inventory.')
            valid_refs={row.ref.strip('/').removeprefix('code/') for row in rows
                if row is not None and getattr(row,'ref','')}
            refs.update(valid_refs)
            if len(rows)<100 or not valid_refs:break
        else:raise ValueError('Owned notebook inventory exceeded bounded pages.')
        for ref in sorted(refs):
            # Kaggle limits status requests even for old notebooks. Pace the
            # complete scan; any 429 still aborts before a GPU write.
            time.sleep(4)
            write_json(folder/'job_scan.json',{'checked_utc':datetime.now(timezone.utc).isoformat(),
                'checked_count':len(states),'total_count':len(refs),'checking':ref,'states':states})
            try:state=api.kernels_status(ref).status.name.upper()
            except requests.HTTPError as error:
                if (ref not in required and error.response is not None
                        and error.response.status_code==404):
                    # Owned legacy notebooks without a saved execution have no
                    # session-status endpoint. Record that limitation explicitly;
                    # never relax source-job checks or allow a GPU reservation.
                    states[ref]='NO_ACCESSIBLE_SAVED_SESSION_HTTP404';continue
                raise
            states[ref]=state
            if state in ACTIVE:raise ValueError('An owned saved Kaggle run is active; no additional dispatch.')
        saved=get_saved(api,SOURCE_KERNEL)
        if saved['metadata']['currentVersionNumber']!=2 or states[SOURCE_KERNEL]!='COMPLETE':
            raise ValueError('Completed grass source version2 required.')
        remaining=remaining_quota(api)
        if reserved_gpu_seconds(api)!=0:raise ValueError('A GPU reservation exists; do not overlap an interactive or saved run.')
        if remaining<CAP+1800:raise ValueError('Insufficient fresh quota for session plus preservation reserve.')
        receipt={'checked_utc':datetime.now(timezone.utc).isoformat(),'status':'dispatch_started',
            'kernel':KERNEL,'version':1,'source':SOURCE,'source_sha256':SOURCE_SHA,'source_folds_sha256':FOLDS_SHA,
            'notebook_sha256':file_sha256(folder/settings['code_file']),
            'source_notebook_sha256':hashlib.sha256(saved['blob']['source'].encode()).hexdigest(),
            'folds':[0,1,2],'arms':['control','shuffle_4','shuffle_10'],
            'remaining_seconds_before_dispatch':remaining,'cap_seconds':CAP,'job_states':states,
            'automatic_retries':False,'competition_submission':False,'new_ssl':False}
        shutil.copy2(folder/settings['code_file'],folder/'submitted_block_shuffle_experiment.ipynb')
        write_json(folder/'launch.json',receipt)  # Write before any remote mutation.
        try:
            response=api.kernels_push(str(folder),timeout=CAP,acc='NvidiaRtxPro6000')
            receipt['acknowledgement']={'ref':response.ref,'version':response.version_number,
                'kernel_id':response.kernel_id,'error_present':bool(response.error)}
            write_json(folder/'launch.json',receipt)
        except Exception as error:
            receipt['ambiguous_write_error_type']=type(error).__name__;write_json(folder/'launch.json',receipt)
        # Even an unexpected ack must be reconciled rather than blindly resent.
        return reconcile(folder,api)


def observe(root,api):
    root=Path(root);folder=root/'outputs/block_shuffle_launch'
    receipt=json.loads((folder/'launch.json').read_text())
    if receipt.get('status')!='submitted_exact_source_readback_verified_actual_training_unverified':
        raise ValueError('Verified submitted identity required before observation.')
    source=get_saved(api,KERNEL)
    if source['metadata']['currentVersionNumber']!=1:raise ValueError('Saved source changed; stop observer.')
    state=api.kernels_status(KERNEL).status.name.upper()
    log=api.kernels_logs(KERNEL) or ''
    (folder/'execution.log').write_text(log)
    status={'checked_utc':datetime.now(timezone.utc).isoformat(),'state':state,'version':1,'log_available':bool(log)}
    if state not in ACTIVE:
        api.kernels_output(KERNEL,path=str(folder/'results'),force=False,quiet=True,
            file_pattern=r'.*(\.json|\.jsonl|\.csv|\.md|\.svg|\.png)$|.*/(selected|latest|best_diagnostic)\.pt$|.*/paper_results\.zip$',page_size=100)
        if get_saved(api,KERNEL)['metadata']['currentVersionNumber']!=1:raise ValueError('Output source changed during collection.')
        report_path=folder/'results/final_one_run/reports/paired_comparisons.json'
        status.update(outputs_collected=True,report_available=report_path.is_file())
    write_json(folder/'status.json',status)
    return status


def watch(root,api):
    folder=Path(root)/'outputs/block_shuffle_launch'
    with (folder/'watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline=time.monotonic()+CAP+1200
        while time.monotonic()<deadline:
            status=observe(root,api)
            if status['state'] not in ACTIVE:return status
            time.sleep(60)
    return {'status':'observer_deadline_no_retry'}


def start_observer(root):
    folder=Path(root)/'outputs/block_shuffle_launch'
    with (folder/'observer.log').open('a') as stream:
        process=subprocess.Popen([sys.executable,__file__,'--watch'],cwd=root,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(folder/'process.json',{'pid':process.pid,'poll_seconds':60,
        'automatic_gpu_retries':False,'automatic_code_repair':False,'agent_wakeup':False})
    return process.pid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--observe',action='store_true');mode.add_argument('--reconcile',action='store_true');mode.add_argument('--watch',action='store_true')
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    try:
        api=bounded_api()
        result=(watch(root,api) if args.watch else observe(root,api) if args.observe else reconcile(root/'outputs/block_shuffle_launch',api) if args.reconcile else launch(root,api))
        print(json.dumps(result),flush=True)
    except Exception as error:
        safe={'stopped_error_type':type(error).__name__,'no_retry':True}
        if isinstance(error,requests.HTTPError) and error.response is not None:
            safe['http_status']=error.response.status_code
        print(json.dumps(safe),flush=True)
        raise SystemExit(1)


if __name__=='__main__':main()
