"""Exact same actual-GPU release regression is RED before fix, GREEN afterward."""
import json,os
from pathlib import Path
import pytest
@pytest.fixture(scope='module')
def proof():
 p=os.environ.get('DINO_RESUME_GPU_PROOF')
 if not p:pytest.skip('Requires actual saved RTX/FP8 bounded diagnostic evidence.')
 return json.loads(Path(p).read_text())
@pytest.mark.parametrize('check',['R0','R1','R2','R3','R4','R5'])
def test_actual_gpu_resume_exact(proof,check):
 assert 'RTX PRO 6000' in proof['environment']['gpu']
 assert proof['full_training_launched'] is False and proof['F3_enabled'] is False
 assert proof['exact_tolerance']==0.
 assert proof['checks'][check]['passed'], json.dumps(proof['checks'][check])
