"""Fresh self-contained dual-study notebook; smoke/full variants retain same code."""
import ast,base64,hashlib,io,json,subprocess,zipfile
from pathlib import Path
KERNEL='meghmayur1186/img2biomass-final-dual-jepa-dino'
SOURCE='meghmayur1186/img2biomass-grass-jepa-frozen-heads/2'

def cell(source,code=True):
 if code:
  ast.parse(source)
  source='try:\n'+''.join('    '+line+'\n' for line in source.splitlines())+"except Exception as error:\n    if 'ROOT' in globals() and 'write' in globals():\n        write(ROOT/'STATUS.json',{'status':'incomplete','stage':'notebook_cell','error_class':type(error).__name__})\n    raise\n"
 result={'cell_type':'code' if code else 'markdown','metadata':{},'source':source.splitlines(keepends=True)}
 if code:ast.parse(source);result.update(execution_count=None,outputs=[])
 return result

def build(root,output,mode='smoke',remaining_seconds=None):
 root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
 if remaining_seconds is None:remaining_seconds=0. # Test-only build; dispatcher rejects insufficient declaredbudget.
 repo=root/'outputs/final_dual_study/offline/dinov3_reference';metadata=root/'outputs/final_dual_study/offline/dinov3_metadata.json'
 archive=io.BytesIO()
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted((root/'biomass_jepa').rglob('*')):
   if p.is_file() and (p.suffix=='.py' or p.name in ('LICENSE','SOURCE.json')):z.write(p,str(p.relative_to(root)))
  for p in sorted(repo.rglob('*')):
   if p.is_file() and '.git' not in p.relative_to(repo).parts and (p.suffix=='.py' or p.name.upper().startswith('LICENSE')):z.write(p,'dinov3_reference/'+str(p.relative_to(repo)))
  z.write(root/'outputs/downloads_2026_10_09/capstone_v20.ipynb','contract/original_capstone_v20.ipynb')
  z.write(root/'outputs/final_dual_study/provenance/SOURCE_BASELINE.json','contract/SOURCE_BASELINE.json')
  z.write(metadata,'dinov3_metadata.json');z.write(root/'tools/install_gpu_deps.py','tools/install_gpu_deps.py')
  for name in ['test_final_dual.py','test_final_dino.py','test_final_figures.py']:z.write(root/'tests'/name,'tests/'+name)
  for name in ['CODEX_IMPLEMENTATION_CONTRACT.md','GPT6_MEDIUM_INDEPENDENT_VERIFIER_PROTOCOL.md']:z.write(Path('/workspace/attachments/dual_study_handoff')/name,'contract/'+name)
 package_sha=hashlib.sha256(archive.getvalue()).hexdigest();payload=base64.b64encode(archive.getvalue()).decode()
 bootstrap=f'''import base64,io,os,sys,zipfile,subprocess,hashlib,json
from pathlib import Path
WORK=Path('/kaggle/working');CODE=WORK/'final_dual_code'
CODE.mkdir(exist_ok=True)
package=base64.b64decode('{payload}')
assert hashlib.sha256(package).hexdigest()=={package_sha!r}
zipfile.ZipFile(io.BytesIO(package)).extractall(CODE)
os.chdir(CODE);sys.path.insert(0,str(CODE));os.environ['PYTHONPATH']=str(CODE)
def mounted(slug,owner):
 roots=[Path('/kaggle/input')/slug,Path('/kaggle/input/datasets')/owner/slug,Path('/kaggle/input/notebooks')/owner/slug,Path('/kaggle/input/kernels')/owner/slug]
 found=[p for p in roots if p.is_dir()]
 if len(found)!=1:raise ValueError('Missing/ambiguous offline input '+slug)
 return found[0]
def unique(root,pattern):
 matches=list(root.rglob(pattern))
 if len(matches)!=1:raise ValueError('Missing/ambiguous mounted file '+pattern)
 return matches[0]
def run(*args):subprocess.run([sys.executable,*map(str,args)],check=True,cwd=CODE)
wheels=mounted('img2biomass-offline-deps','meghmayur1186')
wheel=unique(wheels,'timm-1.0.30-*.whl')
run('tools/install_gpu_deps.py','--wheels',wheel.parent)
'''
 config=f'''from biomass_jepa.final_dual_run import write,CONTRACT
from biomass_jepa.official_heads import feature_code_revision
ROOT=WORK/'final_dual_study'
MODE={mode!r}
write(ROOT/'provenance/source_manifest.json',{{'contract':CONTRACT,'embedded_package_sha256':{package_sha!r},'package_code_sha256':feature_code_revision(),'baseline_commit':{subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!r},'mode':MODE}})
source=mounted('img2biomass-grass-jepa-frozen-heads','meghmayur1186')
adapted=unique(source,'official_ssl/adapted.pt');folds=unique(source,'prepared/folds.csv')
images=mounted('csiro','dheerajkannaujiya')
dino=unique(mounted('dinov3-vitb16','robiulislamistaken'),'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
'''
 setup=f"""from types import SimpleNamespace
import torch,time,shutil
from biomass_jepa.final_dual_run import audit_inputs,smoke,prepare_full,run_j_arm,run_d_cache,run_d_heads,finish
from biomass_jepa.data import file_sha256
from biomass_jepa.final_figures import report
args=SimpleNamespace(adapted=str(adapted),folds=str(folds),images=str(images),dino=str(dino),dino_repository=str(CODE/'dinov3_reference'),dino_metadata=str(CODE/'dinov3_metadata.json'),output=str(ROOT),remaining_seconds={remaining_seconds!r})
def stage(fn,*a,**kw):
    try:return fn(*a,**kw)
    except Exception as error:
        write(ROOT/'STATUS.json',{{'status':'incomplete','stage':fn.__name__,'error_class':type(error).__name__}})
        raise
write(ROOT/'STATUS.json',{{'status':'prepared','mode':MODE,'contract':CONTRACT}})
frame,identity_audit=stage(audit_inputs,args)
def record_runtime_provenance():
    original_baseline=json.loads((CODE/'contract/SOURCE_BASELINE.json').read_text())
    original_baseline['runtime_inputs']={{'adapted':str(adapted),'folds':str(folds),'images':str(images),'dino':str(dino)}}
    assert file_sha256(CODE/'contract/original_capstone_v20.ipynb')==original_baseline['original_notebook']['sha256']
    write(ROOT/'provenance/SOURCE_BASELINE.json',original_baseline)
    write(ROOT/'provenance/target_identity_audit.json',identity_audit)
    import importlib.metadata
    software={{name:importlib.metadata.version(name) for name in ['torch','torchao','timm','numpy','pandas','Pillow','matplotlib','torchvision']}}
    write(ROOT/'provenance/environment.json',{{'software':software,'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'cuda':torch.version.cuda,'fp8':'eligible_linears','loss':'FP32','masters':'FP32'}})
stage(record_runtime_provenance)
write(ROOT/'provenance/inputs_sha256.json',{{'adapted':file_sha256(adapted),'folds':file_sha256(folds),'dino':file_sha256(dino),'labels':file_sha256(images/'train.csv')}})
write(ROOT/'provenance/CONFIG_LOCK.json',{{'contract':CONTRACT,'folds':[0,1],'epochs':200,'swa':40,'j_slots_per_view':20,'actual_mask_arrays_stored':True,'lambda':.10,'mode':MODE,'quota_budget_lower_bound':args.remaining_seconds,'fresh_authoritative_quota_enforced_by_dispatcher':True}})
"""
 driver_source=(root/'tools/run_final_dual.py').read_text()
 resource_gate_source=ast.get_source_segment(driver_source,next(n for n in ast.parse(driver_source).body if isinstance(n,ast.FunctionDef) and n.name=='conservative_resource_gate'))
 stages=[cell('Contract DUAL-JEPA-DINO-2026-10-09-r1. New notebook; old v20 unchanged. P0 required for full dispatch. Five primary arms, two reused development folds, no competition score.',False),cell(bootstrap),cell(config+setup),cell("run('-m','pytest','tests/test_final_dual.py','tests/test_final_dino.py','tests/test_final_figures.py','-q','-k','not locked_builder')\nrun('-m','biomass_jepa.check_gpu')\n"),cell(resource_gate_source+"\n"+"""proof,gate=stage(smoke,args,frame,ROOT/'_smoke')
from biomass_jepa.final_precision import configure as configure_fp8_final
from biomass_jepa.dinov3_dense import make_head,nondegeneracy_proof
from biomass_jepa.final_head import training_statistics
from biomass_jepa.official_heads import GatedRegressionHead
from biomass_jepa.predictor_specialize import state_digest
from biomass_jepa import train as common
subset=__import__('pandas').concat([frame[frame.fold!=0].iloc[:2],frame[frame.fold==0].iloc[:2]]).reset_index(drop=True)
labels=torch.tensor(subset.loc[subset.fold!=0,['Dry_Green_g','Dry_Dead_g','Dry_Clover_g','GDM_g','Dry_Total_g']].to_numpy(),dtype=torch.float32,device='cuda')
scale,median=training_statistics(labels)
proof['head_reload_checks']={}
for arm in ['control','shuffle','D0','D1','D2']:
    isj=arm in ['control','shuffle']
    folder=ROOT/'_smoke'/arm/('head' if isj else '')
    latest=torch.load(folder/'head_latest.pt',weights_only=True,map_location='cuda')
    featurefile=ROOT/'_smoke'/arm/'features/features.pt' if isj else ROOT/'_smoke/dino_features/features.pt'
    feats=torch.load(featurefile,weights_only=True)['features'][0,:2].cuda()
    def instantiate():
        common.seed_all(42)
        model=(GatedRegressionHead(1280,scale,median) if isj else make_head(arm,scale)).cuda().eval()
        configure_fp8_final(model,torch.device('cuda'))
        return model
    initial=instantiate();initial_sha=state_digest(initial)
    one=instantiate();one.load_state_dict(latest['model'],strict=True)
    two=instantiate();two.load_state_dict(latest['model'],strict=True)
    with torch.no_grad(),common.amp(torch.device('cuda')):
        p0=initial(feats).float();p1=one(feats).float();p2=two(feats).float()
    reload_error=float((p1-p2).abs().max());updated=state_digest(one)!=initial_sha
    if reload_error!=0 or not updated:raise RuntimeError('Actual GPU head update/strict reload proof failed.')
    proof['head_reload_checks'][arm]={'steps':latest['steps'],'initial_state_sha256':initial_sha,'after_state_sha256':state_digest(one),'reload_max_abs_error':reload_error,'prediction_update_max_abs':float((p1-p0).abs().max()),'strict_load':True,'updated':updated}
    del initial,one,two,feats
nd_head=make_head('D2',scale).cuda();nd_precision=configure_fp8_final(nd_head,torch.device('cuda'))
with common.amp(torch.device('cuda')):proof['dino_nondegeneracy_actual_fp8']=nondegeneracy_proof(nd_head,torch.randn(2,2,8,8,768,device='cuda'))
proof['dino_nondegeneracy_actual_fp8']['precision']=nd_precision
del nd_head
write(ROOT/'verifier/gpu_smoke.json',proof)
write(ROOT/'verifier/resource_gate_raw.json',gate)
gate=conservative_resource_gate(proof,args.remaining_seconds)
write(ROOT/'verifier/resource_gate.json',gate)
"""),cell("if MODE=='full' and not gate['passes']:raise RuntimeError('Measured all-primary-branches budget fails; no long training.')\nif MODE=='full':prepare_full(ROOT,frame)\nelse:write(ROOT/'STATUS.json',{'status':'smoke_completed_no_full_training','P0_review':'required','resource_gate':gate})\nprint(gate)\n")]
 for fold,arm in [(0,'control'),(0,'shuffle'),(1,'control'),(1,'shuffle')]:stages.append(cell(f"if MODE=='full':stage(run_j_arm,args,frame,ROOT,{fold},{arm!r})\n"))
 stages += [cell("if MODE=='full':dense_features,dense_manifest=stage(run_d_cache,args,frame,ROOT)\n"),cell("if MODE=='full':stage(run_d_heads,args,frame,ROOT,dense_features)\n"),cell("if MODE=='full':stage(report,ROOT,frame)\n"),cell("if MODE=='full':stage(finish,ROOT,frame)\nprint((ROOT/'STATUS.json').read_text())\nprint('Saved-run checkpoints must be verified on Kaggle Output. GPU sessions are temporary.')\n")]
 notebook={'cells':stages,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
 name='capstone_FINAL_DUAL_JEPA_DINO.ipynb';(output/name).write_text(json.dumps(notebook,indent=1)+'\n')
 settings={'id':KERNEL,'title':'Img2BioMass Final Dual JEPA DINO','code_file':name,'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'machine_shape':'NvidiaRtxPro6000','dataset_sources':['dheerajkannaujiya/csiro','meghmayur1186/img2biomass-offline-deps','robiulislamistaken/dinov3-vitb16'],'competition_sources':['arc-prize-2026-arc-agi-3'],'kernel_sources':[SOURCE]}
 (output/'kernel-metadata.json').write_text(json.dumps(settings,indent=2)+'\n');(output/'package.zip').write_bytes(archive.getvalue())
 from biomass_jepa.official_heads import feature_code_revision
 manifest={'quota_budget_lower_bound':remaining_seconds,'contract':'DUAL-JEPA-DINO-2026-10-09-r1','mode':mode,'notebook_sha256':hashlib.sha256((output/name).read_bytes()).hexdigest(),'embedded_package_sha256':package_sha,'source_code_sha256':feature_code_revision()};(output/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--mode',choices=['smoke','full'],default='smoke');p.add_argument('--remaining-seconds',type=float,required=True);a=p.parse_args();print(json.dumps(build(Path(__file__).resolve().parents[1],a.output,a.mode,a.remaining_seconds)))
