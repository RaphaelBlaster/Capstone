"""Self-contained offline frozen grass-I-JEPA spatial permutation experiment."""
import ast
import base64
import io
import json
from pathlib import Path
import subprocess
import zipfile

KERNEL = 'meghmayur1186/img2biomass-jepa-block-shuffle'
SOURCE = 'meghmayur1186/img2biomass-grass-jepa-frozen-heads/2'
SOURCE_SHA = '6d2b5d9c83423155b4a948822e846caaa9a6bc6be20e3fab13ca06b8e705f43d'
FOLDS_SHA = '5f00abf385baf2f20ba0a0790f97d1bf0e7349524c233ae39494bad56b254f5b'
EXTERNAL_SHA = 'f3de96eba431465bdb151f36cc1f6bdd3b4c90b65c295461da891929cefc713b'
CAP = 21600


def cell(text, code=True):
    out={'cell_type':'code' if code else 'markdown','metadata':{},'source':text.splitlines(keepends=True)}
    if code:
        ast.parse(text);out.update(execution_count=None,outputs=[])
    return out


def build(root):
    root=Path(root)
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted((root/'biomass_jepa').rglob('*')):
            if path.is_file() and (path.suffix=='.py' or path.name in {'LICENSE','SOURCE.json'}):
                z.write(path,str(path.relative_to(root)))
        for name in ['requirements.txt','tools/install_gpu_deps.py','paper/block_shuffle_supplied_spec.md',
                     'paper/block_shuffle_protocol.md']:
            z.write(root/name,name)
    payload=base64.b64encode(archive.getvalue()).decode()
    git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    bootstrap=f'''import base64, io, os, sys, zipfile, json, subprocess, time
from pathlib import Path
STARTED=time.monotonic()
WORK = Path('/kaggle/working')
CODE = WORK/'block_shuffle_runner'
CODE.mkdir(parents=True,exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode('{payload}'))).extractall(CODE)
os.chdir(CODE)
sys.path.insert(0,str(CODE))
os.environ['PYTHONPATH']=str(CODE)
def attached_input(slug,owner):
    base=Path('/kaggle/input')
    candidates=[base/slug,base/'datasets'/owner/slug,base/'notebooks'/owner/slug,base/'kernels'/owner/slug]
    present=[p for p in candidates if p.is_dir()]
    if len(present)!=1:raise ValueError('Missing or ambiguous mounted input: '+slug)
    return present[0]
def run(*args):
    subprocess.run([sys.executable,*map(str,args)],cwd=CODE,check=True)
wheels=attached_input('img2biomass-offline-deps','meghmayur1186')
choices=list(wheels.rglob('timm-1.0.30-*.whl'))
if len(choices)!=1:raise ValueError('One offline wheel directory required.')
run('tools/install_gpu_deps.py','--wheels',choices[0].parent)
import torch
if (not torch.cuda.is_available() or 'rtx pro 6000' not in torch.cuda.get_device_name().lower()
        or torch.cuda.get_device_capability()[0]<10):
    raise RuntimeError('Actual RTX PRO6000 Blackwell required; no training fallback.')
run('-m','biomass_jepa.check_gpu')
'''
    preflight=f'''import shutil, hashlib, importlib.metadata
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
from PIL import Image,ImageOps
from biomass_jepa.data import TARGETS,WEIGHTS,file_sha256,image_table
from biomass_jepa.official_heads import source_model,validate_frame,cache_features,fit_heads
from biomass_jepa.train import write_json
from biomass_jepa.block_shuffle import permutation_seed,shuffle_square_tile
from biomass_jepa.block_shuffle_report import report
from types import SimpleNamespace
FOLDS = [0, 1, 2]
ROOT=WORK/'final_one_run'
ROOT.mkdir(exist_ok=True)
CAP={CAP}
SOURCE_KERNEL={SOURCE!r}
SOURCE_SHA={SOURCE_SHA!r}
SOURCE_FOLDS_SHA={FOLDS_SHA!r}
SOURCE_EXTERNAL_SHA={EXTERNAL_SHA!r}
mounted=attached_input('img2biomass-grass-jepa-frozen-heads','meghmayur1186')
def unique_file(pattern):
    found=list(mounted.rglob(pattern))
    if len(found)!=1:raise ValueError('Missing/ambiguous pinned source artifact: '+pattern)
    return found[0]
adapted=unique_file('official_ssl/adapted.pt')
progress_path=unique_file('official_ssl/progress.json')
source_folds=unique_file('prepared/folds.csv')
external=unique_file('prepared/external.json')
ssl_log=unique_file('official_ssl/metrics.jsonl')
progress=json.loads(progress_path.read_text())
if (file_sha256(adapted)!=SOURCE_SHA or progress['export_sha256']['adapted.pt']!=SOURCE_SHA
        or progress['status']!='completed_steps' or progress['step']!=2400
        or progress['unique_images_trained']!=30663
        or file_sha256(source_folds)!=SOURCE_FOLDS_SHA or file_sha256(external)!=SOURCE_EXTERNAL_SHA):
    raise ValueError('Pinned version2 source, progress or data manifest changed.')
# Strict student encoder state; no SSL call or encoder optimizer is constructed.
model,origin=source_model(adapted,'encoder')
if origin['grass_steps']!=2400 or origin['dim']!=1280 or any(p.requires_grad for p in model.parameters()):
    raise ValueError('Need the complete frozen adapted student encoder.')
del model
csiro=attached_input('csiro','dheerajkannaujiya')
trains=list(csiro.rglob('train.csv'))
if len(trains)!=1:raise ValueError('One CSIRO labeled CSV required.')
csiro=trains[0].parent
frame=pd.read_csv(source_folds,dtype={{'group_id':str}})
frame['absolute_path']=[str(csiro/p) for p in frame.image_path]
if (len(frame)!=357 or frame.group_id.nunique()!=28
        or frame.groupby('group_id').fold.nunique().max()!=1
        or len(frame[frame.fold.isin(FOLDS)])!=167):
    raise ValueError('Original fold cohort changed; never generate new folds.')
labels=image_table(trains[0],csiro).sort_values('image_path').reset_index(drop=True)
if (labels.image_path.tolist()!=frame.image_path.tolist()
        or not np.allclose(labels[TARGETS],frame[TARGETS],rtol=0,atol=1e-8)):
    raise ValueError('Mounted labels disagree with the exact saved fold table.')
folds_csv=ROOT/'prepared/folds.csv';folds_csv.parent.mkdir(exist_ok=True)
frame.to_csv(folds_csv,index=False);validate_frame(folds_csv)
ssl_manifest=json.loads(external.read_text())
if len(ssl_manifest)!=30663 or (set(frame.image_sha256)&{{r['sha256'] for r in ssl_manifest}}):
    raise ValueError('SSL manifest count or exact supervised overlap changed.')
for path in frame.absolute_path:
    with Image.open(path) as raw:
        if ImageOps.exif_transpose(raw).size!=(2000,1000):raise ValueError('Expected complete 2000x1000 original field.')
# Verify pixel conservation/bijection and deterministic variants on an actual field.
with Image.open(frame.iloc[0].absolute_path) as raw:
    tile=ImageOps.exif_transpose(raw).convert('RGB').crop((0,0,1000,1000))
    for grid in (4,10):
        variants=[]
        for variant in range(4):
            seed=permutation_seed(frame.iloc[0].image_sha256,grid,variant,0)
            transformed=shuffle_square_tile(tile,grid,seed)
            if transformed.tobytes()!=shuffle_square_tile(tile,grid,seed).tobytes():raise ValueError('Non-deterministic permutation.')
            for ch in range(3):
                if not np.array_equal(np.bincount(np.asarray(tile)[:,:,ch].ravel(),minlength=256),
                    np.bincount(np.asarray(transformed)[:,:,ch].ravel(),minlength=256)):
                    raise ValueError('Permutation lost pixels.')
            variants.append(hashlib.sha256(transformed.tobytes()).hexdigest())
        if len(set(variants))!=4:raise ValueError('Permutation variant collision.')
if shutil.disk_usage(WORK).free<2*1024**3:raise ValueError('Need2GiB for caches, heads and reporting.')
shutil.copy2(source_folds,ROOT/'prepared/source_folds.csv')
shutil.copy2(ssl_log,ROOT/'historical_ssl_metrics.jsonl')
shutil.copy2(CODE/'paper/block_shuffle_protocol.md',ROOT/'FROZEN_ANALYSIS_PLAN.md')
provenance={{'checked_ist':datetime.now(ZoneInfo('Asia/Calcutta')).isoformat(),
    'source_kernel':SOURCE_KERNEL,'source_sha256':SOURCE_SHA,'source_origin':origin,
    'source_progress':progress,'source_fold_sha256':SOURCE_FOLDS_SHA,'mapped_fold_sha256':file_sha256(folds_csv),
    'source_external_manifest_sha256':SOURCE_EXTERNAL_SHA,'exact_ssl_supervised_overlap':0,
    'near_duplicate_overlap':'Not ruled out; exact SHA audit only.',
    'code_base_git_sha':{git_sha!r},'bundle_files_sha256':{{str(p.relative_to(CODE)):file_sha256(p)
        for p in CODE.rglob('*.py')}},'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),
    'cuda':torch.version.cuda,'software':{{n:importlib.metadata.version(n)
        for n in ['torchao','timm','torchvision','numpy','pandas','scikit-learn','Pillow','matplotlib']}},
    'folds':FOLDS,'images':357,'evaluated_images':167,'complete_oof':False,
    'competition_score':False,'seed':42,'shuffle_probability':.35,'epochs':200,'swa_last_epochs':40,
    'source_weight_updates':False,'validation_shuffle':False,
    'limits':'Reused development protocol, date proxies, one seed; not preregistered or protected confirmation.'}}
write_json(ROOT/'PROVENANCE.json',provenance)
print('BLOCK SHUFFLE PREFLIGHT VERIFIED:',json.dumps(provenance),flush=True)
'''
    experiment='''arms=[('control',0,1.,.5),('shuffle_4',4,1.5,.5),('shuffle_10',10,1.5,.5)]
states=[]
import biomass_jepa.train as common
common.STOP_REQUESTED=False
import signal
signal.signal(signal.SIGTERM,common.interrupt_training)
signal.signal(signal.SIGINT,common.interrupt_training)
def budget(cap_hours):
    remaining=CAP-600-(time.monotonic()-STARTED)
    if remaining<120:raise RuntimeError('Session preservation boundary reached; no retry.')
    return min(cap_hours,remaining/3600)
try:
    for name,grid,cache_hours,head_hours in arms:
        if common.STOP_REQUESTED:break
        cache=ROOT/'features'/name
        args=SimpleNamespace(tiny=False,precision='fp8',folds_csv=str(folds_csv),init=str(adapted),
            representation='encoder',output=str(cache),batch_size=32,workers=2,
            max_hours=budget(cache_hours),shuffle_grid=grid)
        cache_features(args)
        if not (cache/'manifest.json').exists():
            states.append({'arm':name,'status':'incomplete_features'});break
        extra=['--shuffle-cache',str(cache),'--shuffle-prob', '.35'] if grid else []
        # Run the same head CLI in separate processes, preserving identical global RNG initialization.
        run('-m','biomass_jepa.official_heads','fit','--cache',ROOT/'features/control',
            *extra,'--folds-csv',folds_csv,'--output',ROOT/'arms'/name,
            '--precision','fp8','--folds',*FOLDS,'--epochs','200','--swa-last-epochs','40',
            '--batch-size','32','--seed','42','--lr','0.0003','--max-hours',budget(head_hours))
        if not (ROOT/'arms'/name/'metrics.json').exists():
            states.append({'arm':name,'status':'incomplete_heads'});break
        data=pd.read_csv(ROOT/'arms'/name/'oof_predictions.csv',dtype={'group_id':str})
        from biomass_jepa.metrics import scores
        rescored=scores(data[TARGETS].to_numpy(),data[['pred_'+t for t in TARGETS]].to_numpy())
        if not np.isclose(rescored['global_weighted_r2_raw'],
                json.loads((ROOT/'arms'/name/'metrics.json').read_text())['global_weighted_r2_raw'],atol=1e-8,rtol=0):
            raise ValueError('Independent stage score mismatch.')
        states.append({'arm':name,'status':'completed','scores':rescored})
        write_json(ROOT/'ARM_STATUS.json',{'arms':states,'seconds':time.monotonic()-STARTED})
        print('ARM VERIFIED:',json.dumps(states[-1]),flush=True)
finally:
    write_json(ROOT/'ARM_STATUS.json',{'arms':states,'seconds':time.monotonic()-STARTED,
        'checkpoint_resume_only_no_automatic_retry':True})
    signatures={}
    for name,_,_,_ in arms:
        p=ROOT/'features'/name/'manifest.json'
        if p.exists():signatures[name]=json.loads(p.read_text())
    write_json(ROOT/'FEATURES_SIGNATURES.json',signatures)
    # The independent report checks cohorts, scale/init/update pairing and uses identity validation only.
    inventory={str(p.relative_to(ROOT)):{'sha256':file_sha256(p),'bytes':p.stat().st_size}
        for p in ROOT.rglob('*') if p.is_file() and p.name!='CHECKPOINT_INVENTORY.json'
        and (p.suffix=='.pt' or p.name=='PROVENANCE.json')}
    write_json(ROOT/'CHECKPOINT_INVENTORY.json',inventory)
    result=report(ROOT,folds_csv,draws=10000,seed=42,ssl_log=ROOT/'historical_ssl_metrics.jsonl',examples=True)
    print('REPORT:',json.dumps(result),flush=True)
'''
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
        'cells':[cell('# Frozen grass-I-JEPA spatial block reshuffling\nThree paired arms; folds0/1/2 (167/357 development images), one seed.\nNo SSL updates; frozen adapted student encoder; training-only spatial permutations.\n',False),
                 cell(bootstrap),cell(preflight),cell(experiment),
                 cell('## Preservation\nKeep final_one_run, all selected/latest checkpoints, logs, predictions and source hashes.\nSaved executions retain outputs. Interactive execution requires Quick Save with Always save output; verify the Output tab.\nNo automatic restart or competition submission. Prior results remain separate.\n',False)]}
    meta={'id':KERNEL,'title':'Img2BioMass JEPA Block Shuffle','code_file':'block_shuffle_experiment.ipynb',
        'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_tpu':False,
        'enable_internet':False,'machine_shape':'NvidiaRtxPro6000',
        'dataset_sources':['dheerajkannaujiya/csiro','meghmayur1186/img2biomass-offline-deps'],
        'competition_sources':['arc-prize-2026-arc-agi-3'],'kernel_sources':[SOURCE]}
    return notebook,meta


def prepare(root):
    root=Path(root);out=root/'outputs/block_shuffle_launch';out.mkdir(parents=True,exist_ok=True)
    if (out/'launch.json').exists():raise ValueError('Never rebuild a dispatched notebook.')
    notebook,meta=build(root)
    (out/meta['code_file']).write_text(json.dumps(notebook,indent=1))
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    return out


if __name__=='__main__':print(prepare(Path(__file__).resolve().parents[1]))
