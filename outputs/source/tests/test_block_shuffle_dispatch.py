import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools import run_block_shuffle_experiment as driver
from tools.build_block_shuffle_experiment import KERNEL


@pytest.mark.parametrize('reason',['journal','existing','active','quota','unverified'])
def test_dispatch_stops_without_gpu_write(tmp_path,monkeypatch,reason):
    monkeypatch.setattr(driver.time,'sleep',lambda _:None)
    folder=tmp_path/'outputs/block_shuffle_launch';folder.mkdir(parents=True)
    (folder/'kernel-metadata.json').write_text(json.dumps({'id':KERNEL,'code_file':'x.ipynb'}))
    (folder/'x.ipynb').write_text('{"cells":[]}')
    if reason=='journal':(folder/'launch.json').write_text('{}')
    monkeypatch.setattr(driver,'validate_prepared',lambda *_:None)
    monkeypatch.setattr(driver,'validate_local_evidence',lambda *_:(_ for _ in ()).throw(ValueError('unverified')) if reason=='unverified' else None)
    monkeypatch.setattr(driver,'get_inference_source',lambda *_:SimpleNamespace() if reason=='existing' else None)
    monkeypatch.setattr(driver,'get_saved',lambda *_:{'metadata':{'currentVersionNumber':2},'blob':{'source':'{}'}})
    monkeypatch.setattr(driver,'remaining_quota',lambda _:0 if reason=='quota' else 30000)
    monkeypatch.setattr(driver,'reserved_gpu_seconds',lambda _:0)
    api=SimpleNamespace(config_values={'username':'meghmayur1186'},
        kernels_list=lambda **_:[],
        kernels_status=lambda _:SimpleNamespace(status=SimpleNamespace(name='RUNNING' if reason=='active' else 'COMPLETE')),
        kernels_push=lambda *_,**__:pytest.fail('guard must prevent GPU write'))
    with pytest.raises(ValueError):driver.launch(tmp_path,api)
    assert not (folder/'submitted_block_shuffle_experiment.ipynb').exists()


@pytest.mark.parametrize('changed',[False,True])
def test_accepted_write_readback_never_pushes_again(tmp_path,monkeypatch,changed):
    from tools.build_block_shuffle_experiment import SOURCE
    folder=tmp_path/'outputs/block_shuffle_launch';folder.mkdir(parents=True)
    notebook={'cells':[{'cell_type':'code','source':['print(1)\n']}]}
    path=folder/'submitted_block_shuffle_experiment.ipynb';path.write_text(json.dumps(notebook))
    settings={'dataset_sources':['a','b'],'competition_sources':['arc-prize-2026-arc-agi-3']}
    (folder/'kernel-metadata.json').write_text(json.dumps(settings))
    receipt={'status':'dispatch_started','notebook_sha256':driver.file_sha256(path)}
    (folder/'launch.json').write_text(json.dumps(receipt))
    remote={'cells':[{'cell_type':'code','source':['print(2)\n' if changed else 'print(1)\n']}]}
    actual={'metadata':{'ref':KERNEL,'currentVersionNumber':1,'machineShape':'NvidiaRtxPro6000',
        'enableGpu':True,'isPrivate':True,'enableInternet':False,'enableTpu':False,
        'datasetDataSources':['a','b'],'competitionDataSources':settings['competition_sources'],
        'kernelDataSources':[SOURCE.rsplit('/',1)[0]],'id':123},'blob':{'source':json.dumps(remote)}}
    monkeypatch.setattr(driver,'get_saved',lambda *_:actual)
    api=SimpleNamespace(kernels_push=lambda *_,**__:pytest.fail('Readback must never retry the write'))
    if changed:
        with pytest.raises(ValueError,match='do not retry'):driver.reconcile(folder,api)
        assert json.loads((folder/'launch.json').read_text())==receipt
    else:
        row=driver.reconcile(folder,api)
        assert row['status']=='submitted_exact_source_readback_verified_actual_training_unverified'


@pytest.mark.parametrize('missing',['none','legacy','source','reserved','rate_limited'])
def test_owned_listing_empty_sdk_slots_do_not_create_status_requests(tmp_path,monkeypatch,missing):
    monkeypatch.setattr(driver.time,'sleep',lambda _:None)
    import requests
    folder=tmp_path/'outputs/block_shuffle_launch';folder.mkdir(parents=True)
    (folder/'kernel-metadata.json').write_text(json.dumps({'id':KERNEL,'code_file':'x.ipynb'}))
    (folder/'x.ipynb').write_text('{"cells":[]}')
    monkeypatch.setattr(driver,'validate_prepared',lambda *_:None)
    monkeypatch.setattr(driver,'validate_local_evidence',lambda *_:None)
    monkeypatch.setattr(driver,'get_inference_source',lambda *_:None)
    monkeypatch.setattr(driver,'get_saved',lambda *_:{'metadata':{'currentVersionNumber':2},'blob':{'source':'{}'}})
    monkeypatch.setattr(driver,'remaining_quota',lambda _:30000)
    monkeypatch.setattr(driver,'reserved_gpu_seconds',lambda _:100 if missing=='reserved' else 0)
    monkeypatch.setattr(driver,'reconcile',lambda folder,api:json.loads((folder/'launch.json').read_text()))
    calls=[]
    def status(ref):
        assert ref, 'No status call for an empty SDK slot'
        if missing=='rate_limited':
            raise requests.HTTPError(response=SimpleNamespace(status_code=429))
        if ref.endswith('/legacy') or (missing=='source' and ref==driver.SOURCE_KERNEL):
            raise requests.HTTPError(response=SimpleNamespace(status_code=404))
        return SimpleNamespace(status=SimpleNamespace(name='COMPLETE'))
    api=SimpleNamespace(config_values={'username':'meghmayur1186'},
        kernels_list=lambda **k:([SimpleNamespace(ref='meghmayur1186/legacy')] if missing=='legacy' else [SimpleNamespace(ref='')]*100)
            if k.get('page',1)==1 else [],kernels_status=status,
        kernels_push=lambda *a,**k:calls.append((a,k)) or SimpleNamespace(ref=KERNEL,version_number=1,kernel_id=1,error=None))
    if missing in ('source','reserved','rate_limited'):
        with pytest.raises((ValueError,requests.HTTPError)):driver.launch(tmp_path,api)
        assert not calls
        return
    driver.launch(tmp_path,api)
    assert len(calls)==1
    with pytest.raises(ValueError,match='never launch twice'):driver.launch(tmp_path,api)
    assert len(calls)==1
