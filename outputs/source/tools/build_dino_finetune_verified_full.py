"""Build full wrapper only from independently passed immutable GPU package."""
import ast,base64,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
O=ROOT/'outputs/dino_resume_rca/full_launch';O.mkdir(parents=True,exist_ok=True)
if (O/'launch.json').exists():raise ValueError('Never rebuild submitted full source.')
review=json.loads((ROOT/'outputs/dino_resume_rca/GPU_RESUME_VERDICT.json').read_text());config=json.loads((ROOT/'outputs/dino_finetune/config.json').read_text());blob=(ROOT/'outputs/dino_resume_rca/green/launch/package.zip').read_bytes();sha=hashlib.sha256(blob).hexdigest()
assert review['status']=='PASS' and review['verdict']=='PASS_GPU_RESUME' and review['gpu_fp8_verified'] and review['package_sha256']==sha
assert hashlib.sha256((ROOT/'outputs/dino_finetune/config.json').read_bytes()).hexdigest()==review['config_sha256']
auth={'authorized':True,'mode':'full','package_sha256':sha,'basis':'Latest user: start the training bro, then continue; only after independent PASS_GPU_RESUME','F3_enabled':False}
proof={**review,'config':config}
nb=json.loads((ROOT/'outputs/dino_finetune/notebook/DINO_ANCHORED_FINETUNE.ipynb').read_text());code=''.join(nb['cells'][1]['source']).replace("MODE='prepare'","MODE='full'",1)
code=re.sub(r"PACKAGE_SHA='[^']+'",'PACKAGE_SHA='+repr(sha),code,count=1);code=re.sub(r"blob=base64.b64decode\('[^']+'\)",lambda _: 'blob=base64.b64decode('+repr(base64.b64encode(blob).decode())+')',code,count=1)
needle='require_execution_permission(MODE,PACKAGE_SHA,VERIFICATION,AUTHORIZATION)'
gates=f"VERIFICATION.write_text({json.dumps(review)!r})\nAUTHORIZATION.write_text({json.dumps(auth)!r})\nRUNTIME_PROOF.write_text({json.dumps(proof)!r})\nassert json.loads(RUNTIME_PROOF.read_text())['verdict']=='PASS_GPU_RESUME'\n"
code=code.replace(needle,gates+needle,1);code=code.replace('# This notebook was prepared with enable_gpu=False. Never submitted by this builder.','# Exact passed FP8 package; approved matched F1/F2 full run. F3 disabled.')
ast.parse(code);nb['cells'][1]['source']=code.splitlines(keepends=True);nb['cells'][0]['source']=['Verified deterministic FP8 DINOv3 F1/F2: 20 epochs each, grouped folds0/1, unchanged D2 and L2-SP. Independent PASS_GPU_RESUME bound to exact package. No F3. Progress bars and atomic resumable checkpoints enabled.'];name='DINO_F1_F2_VERIFIED_FULL.ipynb';raw=(json.dumps(nb,indent=1)+'\n').encode();(O/name).write_bytes(raw);(O/'package.zip').write_bytes(blob)
meta=json.loads((ROOT/'outputs/dino_finetune/smoke_launch/kernel-metadata.json').read_text());meta.update(id='meghmayur1186/img2biomass-dino-finetune-f1-f2',title='Img2BioMass DINO Finetune F1 F2',code_file=name);(O/'kernel-metadata.json').write_text(json.dumps(meta,indent=2));(O/'USER_EXECUTION_AUTHORIZATION.json').write_text(json.dumps(auth,indent=2));(O/'GPU_RESUME_VERDICT.json').write_text(json.dumps(review,indent=2));m={'notebook_sha256':hashlib.sha256(raw).hexdigest(),'package_sha256':sha,'trainer_sha256':review['trainer_sha256'],'mode':'full','config':config,'folds':[0,1],'arms':['F1','F2'],'F3_enabled':False,'execution_cap_seconds':10800,'estimated_budget_with_margin_seconds':5604.395956704,'independent_resume_verdict':'PASS_GPU_RESUME'};(O/'build_manifest.json').write_text(json.dumps(m,indent=2));print(json.dumps(m))
