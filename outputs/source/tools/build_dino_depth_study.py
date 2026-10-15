"""Build a distinct immutable offline RTX depth study; never rewrite old notebooks."""
import ast
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path
from tools.build_dino_finetune import build

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/dino_depth_study/launch'
KERNEL='meghmayur1186/img2biomass-dino-depth-2-4-8'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'launch.json').exists():raise ValueError('Never rebuild submitted depth source.')
    # Reuse pinned dependencies/source identities; original builder outputs stay intact.
    template=ROOT/'outputs/dino_depth_study/template'
    build(ROOT,template)
    with zipfile.ZipFile(template/'package.zip') as z:files={n:z.read(n) for n in z.namelist() if n!='PACKAGE_FILES_SHA256.json'}
    files['tests/test_dino_depth.py']=(ROOT/'tests/test_dino_depth.py').read_bytes()
    files['DEPTH_PROTOCOL.md']=(ROOT/'paper/dino_depth_protocol.md').read_bytes()
    files['PACKAGE_FILES_SHA256.json']=json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in files.items()},sort_keys=True,indent=2).encode()
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n,b in sorted(files.items()):
            info=zipfile.ZipInfo(n,date_time=(2026,10,11,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,b)
    blob=stream.getvalue();sha=hashlib.sha256(blob).hexdigest();(OUT/'package.zip').write_bytes(blob)
    code=f'''import base64,hashlib,io,json,os,subprocess,sys,time,zipfile
from pathlib import Path
START=time.monotonic()
PACKAGE_SHA={sha!r}
WORK=Path('/kaggle/working');CODE=WORK/'dino_depth_code';CODE.mkdir(exist_ok=True)
blob=base64.b64decode({base64.b64encode(blob).decode()!r})
assert hashlib.sha256(blob).hexdigest()==PACKAGE_SHA
zipfile.ZipFile(io.BytesIO(blob)).extractall(CODE)
for name,expected in json.loads((CODE/'PACKAGE_FILES_SHA256.json').read_text()).items():
    assert hashlib.sha256((CODE/name).read_bytes()).hexdigest()==expected
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PYTHONPATH']=str(CODE);sys.path.insert(0,str(CODE));os.chdir(CODE)
VERIFICATION=WORK/'STATIC_VERDICT.json'
AUTHORIZATION=WORK/'USER_EXECUTION_AUTHORIZATION.json'
# Builder binds the independent review in a separate finalization step before dispatch.
REVIEW_PLACEHOLDER
auth={{'authorized':True,'mode':'full','package_sha256':PACKAGE_SHA,'depths':[2,4,8],'basis':'User11Oct: no convolution; run training at different unfreezing depths fast; conditional full training after all-depth runtime gates','F3_enabled':False}}
AUTHORIZATION.write_text(json.dumps(auth))
review=json.loads(VERIFICATION.read_text())
assert review['status']=='PASS' and review['package_sha256']==PACKAGE_SHA and review['depths']==[2,4,8]
def mounted(slug):
    matches=[p for p in Path('/kaggle/input').rglob(slug) if p.is_dir()]
    if len(matches)!=1:raise ValueError('Missing/ambiguous offline mount '+slug)
    return matches[0]
def unique(root,pattern):
    choices=list(root.rglob(pattern))
    if len(choices)!=1:raise ValueError('Missing/ambiguous input '+pattern)
    return choices[0]
def run(*args):
    remaining=21600-(time.monotonic()-START)
    if remaining<=0:raise TimeoutError('Fixed notebook execution cap exhausted.')
    subprocess.run([sys.executable,*map(str,args)],check=True,cwd=CODE,timeout=remaining)
wheels=unique(mounted('img2biomass-offline-deps'),'timm-1.0.30-*.whl')
run('tools/install_gpu_deps.py','--wheels',wheels.parent)
run('-m','pytest','tests/test_dino_finetune.py','tests/test_dino_depth.py','-q')
run('-m','biomass_jepa.check_gpu')
source=unique(mounted('img2biomass-final-dual-jepa-dino'),'provenance/folds.csv').parent.parent
images=mounted('csiro');dino=unique(mounted('dinov3-vitb16'),'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
run('-m','biomass_jepa.dino_depth_study','--source',source,'--images',images,'--repository',CODE/'dinov3_reference','--checkpoint',dino,'--metadata',CODE/'dinov3_metadata.json','--lock',CODE/'input_lock.json','--output',WORK/'dino_depth_study','--verification',VERIFICATION,'--authorization',AUTHORIZATION,'--embedded-package-sha',PACKAGE_SHA,'--execution-budget-seconds',21600-(time.monotonic()-START))
print((WORK/'dino_depth_study/STATUS.json').read_text())
'''
    ast.parse(code)
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}},'cells':[{'cell_type':'markdown','metadata':{},'source':['Matched convolution-free official DINOv3 B16 D2 depth2/4/8;20epochs/folds0/1. Independent static PASS required; all-depth actual FP8/parity/resume/resource gates precede full training. F3disabled. Original results preserved.']},{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(keepends=True)}]}
    name='DINO_DEPTH_2_4_8.ipynb';(OUT/name).write_text(json.dumps(nb,indent=1)+'\n')
    meta=json.loads((ROOT/'outputs/dino_finetune/smoke_launch/kernel-metadata.json').read_text())
    meta.update(id=KERNEL,title='Img2BioMass DINO Depth 2 4 8',code_file=name)
    (OUT/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    manifest={'package_sha256':sha,'notebook_sha256':hashlib.sha256((OUT/name).read_bytes()).hexdigest(),'depths':[2,4,8],'folds':[0,1],'epochs':20,'F3_enabled':False,'no_convolution':True,'execution_cap_seconds':21600,'conditional_full_on_all_runtime_gates':True,'status':'PREPARED_REVIEW_PENDING'}
    (OUT/'build_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))

if __name__=='__main__':main()
