import ast,base64,io,json,re,zipfile
from pathlib import Path
from tools.build_block_shuffle_experiment import build, CAP, KERNEL


def test_standalone_notebook_freezes_protocol_and_inputs():
    root=Path(__file__).resolve().parents[1]
    notebook,meta=build(root)
    source='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    for c in notebook['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    assert meta['id']==KERNEL and meta['enable_internet'] is False and meta['is_private'] is True
    assert meta['machine_shape']=='NvidiaRtxPro6000'
    assert meta['dataset_sources']==['dheerajkannaujiya/csiro','meghmayur1186/img2biomass-offline-deps']
    assert meta['kernel_sources']==['meghmayur1186/img2biomass-grass-jepa-frozen-heads/2']
    assert 'FOLDS = [0, 1, 2]' in source and "'--shuffle-prob', '.35'" in source
    assert '2400' in source and 'source_model' in source
    assert "'biomass_jepa.official_train'" not in source and "'capstone/20'" not in source
    assert 'PROVENANCE.json' in source and 'FROZEN_ANALYSIS_PLAN.md' in source
    assert '600' in source and CAP==21600
    payload=re.search(r"base64.b64decode\('([^']+)'\)",source).group(1)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as z:
        for f in ['biomass_jepa/block_shuffle.py','biomass_jepa/data.py','biomass_jepa/official_heads.py',
                  'biomass_jepa/block_shuffle_report.py']:
            assert z.read(f)==(root/f).read_bytes()
        assert all(not n.endswith('.pt') for n in z.namelist())
