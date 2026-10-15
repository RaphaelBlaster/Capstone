"""Deterministic offline implementation package. Default notebook is CPU preparation only."""
import ast
import base64
import hashlib
import inspect
import io
import json
import zipfile
from pathlib import Path

KERNEL='meghmayur1186/img2biomass-dino-anchored-finetune'
SOURCE='meghmayur1186/img2biomass-final-dual-jepa-dino/2'


def require_execution_permission(mode,package_sha,verification,authorization):
    if mode=='prepare':return
    review=json.loads(Path(verification).read_text());auth=json.loads(Path(authorization).read_text())
    if review.get('status')!='PASS' or review.get('package_sha256')!=package_sha:raise ValueError('Exact embedded source independent PASS required.')
    if auth.get('authorized') is not True or auth.get('mode')!=mode or auth.get('package_sha256')!=package_sha:raise ValueError('New explicit exact-package execution authorization required.')


def build(root,output,mode='prepare'):
    if mode!='prepare':raise ValueError('Only preparation notebooks authorized for this task.')
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    prior=root/'outputs/final_dual_study/full_results/final_dual_study'
    inputs=['provenance/folds.csv','dinov3/cache_manifest.json']
    for fold in (0,1):inputs += [f'dinov3/fold_{fold}/D2/head_selected.pt',f'dinov3/fold_{fold}/D2/metrics.json']
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    lock={'source':SOURCE,'inputs':{name:sha(prior/name) for name in inputs}}
    files={}
    for p in sorted((root/'biomass_jepa').rglob('*.py')):files[str(p.relative_to(root))]=p.read_bytes()
    for p in sorted((root/'biomass_jepa').rglob('*')):
        if p.is_file() and p.name in ('LICENSE','SOURCE.json'):files[str(p.relative_to(root))]=p.read_bytes()
    repo=root/'outputs/final_dual_study/offline/dinov3_reference'
    for p in sorted(repo.rglob('*')):
        if p.is_file() and '.git' not in p.parts and (p.suffix=='.py' or p.name.upper().startswith('LICENSE')):files['dinov3_reference/'+str(p.relative_to(repo))]=p.read_bytes()
    for src,dest in [('outputs/final_dual_study/offline/dinov3_metadata.json','dinov3_metadata.json'),('tools/install_gpu_deps.py','tools/install_gpu_deps.py'),('tests/test_dino_finetune.py','tests/test_dino_finetune.py'),('outputs/dino_finetune/config.json','config.json')]:files[dest]=(root/src).read_bytes()
    files['input_lock.json']=json.dumps(lock,sort_keys=True,indent=2).encode()
    files['PACKAGE_FILES_SHA256.json']=json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in files.items()},sort_keys=True,indent=2).encode()
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,content in sorted(files.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,10,10,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,content)
    blob=stream.getvalue();package_sha=hashlib.sha256(blob).hexdigest()
    gate=inspect.getsource(require_execution_permission)
    code=f'''import ast,base64,hashlib,io,json,os,subprocess,sys,zipfile
from pathlib import Path
MODE='prepare'
PACKAGE_SHA={package_sha!r}
VERIFICATION=Path('/kaggle/working/P0_STATIC_VERDICT.json')
AUTHORIZATION=Path('/kaggle/working/USER_EXECUTION_AUTHORIZATION.json')
RUNTIME_PROOF=Path('/kaggle/working/P0_GPU_VERDICT.json')
{gate}
require_execution_permission(MODE,PACKAGE_SHA,VERIFICATION,AUTHORIZATION)
WORK=Path('/kaggle/working');CODE=WORK/'dino_finetune_code';CODE.mkdir(exist_ok=True)
blob=base64.b64decode({base64.b64encode(blob).decode()!r})
assert hashlib.sha256(blob).hexdigest()==PACKAGE_SHA
zipfile.ZipFile(io.BytesIO(blob)).extractall(CODE)
for name,expected in json.loads((CODE/'PACKAGE_FILES_SHA256.json').read_text()).items():
    assert hashlib.sha256((CODE/name).read_bytes()).hexdigest()==expected
os.chdir(CODE);sys.path.insert(0,str(CODE));os.environ['PYTHONPATH']=str(CODE)
def mounted(slug):
    matches=[p for p in Path('/kaggle/input').rglob(slug) if p.is_dir()]
    if len(matches)!=1:raise ValueError('Missing/ambiguous offline mount '+slug)
    return matches[0]
def unique(root,pattern):
    choices=list(root.rglob(pattern))
    if len(choices)!=1:raise ValueError('Missing/ambiguous input '+pattern)
    return choices[0]
def run(*args):subprocess.run([sys.executable,*map(str,args)],check=True,cwd=CODE)
# This notebook was prepared with enable_gpu=False. Never submitted by this builder.
if MODE!='prepare':
    wheels=unique(mounted('img2biomass-offline-deps'),'timm-1.0.30-*.whl')
    run('tools/install_gpu_deps.py','--wheels',wheels.parent)
    run('-m','pytest','tests/test_dino_finetune.py','-q')
    run('-m','biomass_jepa.check_gpu')
source=unique(mounted('img2biomass-final-dual-jepa-dino'),'provenance/folds.csv').parent.parent
images=mounted('csiro')
dino=unique(mounted('dinov3-vitb16'),'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
arguments=['-m','biomass_jepa.dino_finetune','--source',source,'--images',images,'--repository',CODE/'dinov3_reference','--checkpoint',dino,'--metadata',CODE/'dinov3_metadata.json','--lock',CODE/'input_lock.json','--output',WORK/('dino_finetune_'+MODE),'--mode',MODE,'--embedded-package-sha',PACKAGE_SHA]
if MODE!='prepare':arguments+=['--verification',VERIFICATION,'--authorization',AUTHORIZATION]
if MODE=='full':arguments+=['--runtime-proof',RUNTIME_PROOF]
run(*arguments)
print((WORK/('dino_finetune_'+MODE)/'STATUS.json').read_text())
'''
    ast.parse(code)
    filename='DINO_ANCHORED_FINETUNE.ipynb'
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python'}},'cells':[{'cell_type':'markdown','metadata':{},'source':['PREPARED ONLY. Matched F1/F2 D2 continuation from complete photographs. Default mode validates inputs without GPU or training. F3 is separate preparation only. Changing mode requires exact-package independent PASS plus new user execution authorization; full training also requires independently verified RTX/FP8 smoke. No new score or novelty claimed.']},{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(keepends=True)}]}
    (output/filename).write_text(json.dumps(nb,indent=1)+'\n');(output/'package.zip').write_bytes(blob);(output/'input_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    settings={'id':KERNEL,'title':'Img2BioMass DINO Anchored Finetune','code_file':filename,'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':False,'enable_tpu':False,'enable_internet':False,'machine_shape':'NvidiaRtxPro6000','dataset_sources':['meghmayur1186/img2biomass-offline-deps','dheerajkannaujiya/csiro','robiulislamistaken/dinov3-vitb16'],'competition_sources':['arc-prize-2026-arc-agi-3'],'kernel_sources':[SOURCE]}
    (output/'kernel-metadata.json').write_text(json.dumps(settings,indent=2)+'\n')
    manifest={'status':'PREPARED_NOT_SUBMITTED','mode':'prepare','notebook_sha256':sha(output/filename),'package_sha256':package_sha,'source_file_hashes':{n:hashlib.sha256(b).hexdigest() for n,b in files.items()},'input_lock':lock,'GPU_enabled':False,'arms':['F1','F2'],'F3_enabled':False,'no_dispatch_implemented':True}
    (output/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest

if __name__=='__main__':
    m=build(Path(__file__).resolve().parents[1],'outputs/dino_finetune/notebook');print(json.dumps({k:m[k] for k in ['status','notebook_sha256','package_sha256']}))
