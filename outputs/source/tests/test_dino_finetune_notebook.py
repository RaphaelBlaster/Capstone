"""Offline builder contracts: inert default, immutable sources and explicit gates."""
import ast
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest


def fixture_root(tmp_path):
    root=tmp_path/'repo'
    files={
        'biomass_jepa/dino_finetune.py':'CONFIG = {"arms": ["F1", "F2"]}\n',
        'biomass_jepa/__init__.py':'',
        'tools/install_gpu_deps.py':'# offline installer\n',
        'tests/test_dino_finetune.py':'def test_example(): assert True\n',
        'outputs/final_dual_study/offline/dinov3_reference/hubconf.py':'# official fixture\n',
        'outputs/final_dual_study/offline/dinov3_reference/LICENSE.md':'license',
        'outputs/final_dual_study/offline/dinov3_metadata.json':'{"checkpoint_sha256":"official"}',
        'outputs/dino_finetune/config.json':'{"arms":["F1","F2"],"epochs":20}',
    }
    prior='outputs/final_dual_study/full_results/final_dual_study/'
    for name in ['provenance/folds.csv','dinov3/cache_manifest.json']:
        files[prior+name]=name
    for fold in (0,1):
        for name in ('head_selected.pt','metrics.json'):
            files[prior+f'dinov3/fold_{fold}/D2/{name}']=f'{fold}/{name}'
    for name,content in files.items():
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
    return root


def test_builder_exists():
    assert (Path(__file__).resolve().parents[1]/'tools/build_dino_finetune.py').exists()


def test_default_prepare_embeds_verified_code_and_strict_locks(tmp_path):
    from tools.build_dino_finetune import build
    root=fixture_root(tmp_path);out=tmp_path/'output';manifest=build(root,out)
    nb=json.loads((out/'DINO_ANCHORED_FINETUNE.ipynb').read_text())
    code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    ast.parse(code)
    assert "MODE='prepare'" in code
    assert code.index('require_execution_permission(')<code.index("run('tools/install_gpu_deps.py'")
    assert 'biomass_jepa.dino_finetune' in code
    assert 'biomass_jepa.check_gpu' in code
    meta=json.loads((out/'kernel-metadata.json').read_text())
    assert not meta['enable_internet'] and not meta['enable_gpu']
    assert meta['is_private'] and meta['machine_shape']=='NvidiaRtxPro6000'
    assert meta['kernel_sources']==['meghmayur1186/img2biomass-final-dual-jepa-dino/2']
    blob=(out/'package.zip').read_bytes()
    assert hashlib.sha256(blob).hexdigest()==manifest['package_sha256']
    z=zipfile.ZipFile(io.BytesIO(blob));names=z.namelist()
    assert 'biomass_jepa/dino_finetune.py' in names
    assert 'dinov3_reference/LICENSE.md' in names
    assert 'config.json' in names and 'input_lock.json' in names
    assert not any(n.endswith('.pt') for n in names)
    lock=json.loads(z.read('input_lock.json'))
    assert len(lock['inputs'])==6
    for name,expected in lock['inputs'].items():
        assert hashlib.sha256((root/'outputs/final_dual_study/full_results/final_dual_study'/name).read_bytes()).hexdigest()==expected
    assert not any('features.pt' in p for p in lock['inputs'])


def test_permission_gate_rejects_missing_or_mismatched_review_and_authorization(tmp_path):
    from tools.build_dino_finetune import require_execution_permission
    missing=tmp_path/'missing'
    require_execution_permission('prepare','abc',missing,missing)
    with pytest.raises((ValueError,FileNotFoundError)):
        require_execution_permission('full','abc',missing,missing)
    review=tmp_path/'review.json';auth=tmp_path/'authorization.json'
    review.write_text(json.dumps({'status':'PASS','package_sha256':'abc'}))
    auth.write_text(json.dumps({'authorized':True,'mode':'smoke','package_sha256':'abc'}))
    require_execution_permission('smoke','abc',review,auth)
    with pytest.raises(ValueError):require_execution_permission('full','abc',review,auth)
    auth.write_text(json.dumps({'authorized':True,'mode':'full','package_sha256':'wrong'}))
    with pytest.raises(ValueError):require_execution_permission('full','abc',review,auth)
    auth.write_text(json.dumps({'authorized':True,'mode':'full','package_sha256':'abc'}))
    review.write_text(json.dumps({'status':'UNVERIFIED','package_sha256':'abc'}))
    with pytest.raises(ValueError):require_execution_permission('full','abc',review,auth)


def test_build_is_deterministic_and_rejects_training_default(tmp_path):
    from tools.build_dino_finetune import build
    root=fixture_root(tmp_path)
    a=build(root,tmp_path/'one');b=build(root,tmp_path/'two')
    assert a['package_sha256']==b['package_sha256']
    assert a['notebook_sha256']==b['notebook_sha256']
    with pytest.raises(ValueError):build(root,tmp_path/'bad',mode='full')
