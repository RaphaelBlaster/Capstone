import ast
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_supervised_notebook_is_separate_offline_and_interleaved():
    from tools.build_supervised_shuffle_experiment import build
    notebook,meta=build(ROOT)
    code='\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type']=='code')
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            ast.parse(''.join(cell['source']))
    assert meta['id']=='meghmayur1186/img2biomass-jepa-supervised-shuffle'
    assert meta['enable_internet'] is False
    assert meta['machine_shape']=='NvidiaRtxPro6000'
    assert meta['competition_sources']==['arc-prize-2026-arc-agi-3']
    assert meta['kernel_sources']==['meghmayur1186/img2biomass-grass-jepa-frozen-heads/2']
    assert "FOLDS=[0,1,2]" in code
    assert "for fold in FOLDS:" in code
    assert "for name,grid,probability in ARMS:" in code
    assert "'--run-fold',fold" in code
    assert "'--encoder-lr','5e-6'" in code
    assert "'--head-lr','1e-4'" in code
    assert "'--shuffle-prob',probability" in code
    assert "('shuffle_4',4,.2)" in code
    assert code.index('STARTED=time.monotonic()') < code.index('install_gpu_deps.py')
    assert code.index("CHECKPOINT_INVENTORY.json'") < code.index('result=report(')
    assert 'REPORT_FAILURE.json' in code
    assert 'remaining_stage_seconds' in code
    assert 'source_weight_updates' in code
    assert "'--folds',*FOLDS" in code
    assert "'--epochs','200'" in code
    assert 'capstone/20' not in code


def test_prepare_cannot_overwrite_dispatched_supervised_notebook(tmp_path):
    from tools.build_supervised_shuffle_experiment import prepare
    folder=tmp_path/'outputs/supervised_shuffle_launch'
    folder.mkdir(parents=True)
    (folder/'launch.json').write_text('{}')
    with pytest.raises(ValueError,match='dispatched'):
        prepare(tmp_path)


def test_supervised_notebook_attaches_native_auxiliary_labels_without_mapping():
    from tools.build_supervised_shuffle_experiment import build
    notebook,meta=build(ROOT)
    code='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    assert 'pushpakbhoge/grass-clover-datasets/1' in meta['dataset_sources']
    assert 'bed4efd6f40fec31f39f5fa0f4273097e361fc7d3a3c75d51749942518fcfceb' in code
    assert "'--aux-folds-csv',aux_csv" in code
    assert "'--aux-weight','.25'" in code
    assert 'prepare_native_auxiliary' in code
    assert '261' in code
