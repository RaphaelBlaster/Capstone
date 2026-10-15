"""Prepare a fresh offline notebook: supervised encoder/head online block permutation."""
import base64
import io
import json
import subprocess
import zipfile
from pathlib import Path

from tools.build_block_shuffle_experiment import cell, SOURCE, SOURCE_SHA, FOLDS_SHA, EXTERNAL_SHA

KERNEL='meghmayur1186/img2biomass-jepa-supervised-shuffle'
CAP=21600


def build(root):
    root=Path(root)
    bundle=io.BytesIO()
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((root/'biomass_jepa').rglob('*')):
            if path.is_file() and (path.suffix=='.py' or path.name in {'LICENSE','SOURCE.json'}):
                archive.write(path,str(path.relative_to(root)))
        for name in ['requirements.txt','tools/install_gpu_deps.py','paper/supervised_shuffle_protocol.md']:
            archive.write(root/name,name)
    payload=base64.b64encode(bundle.getvalue()).decode()
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    bootstrap=f'''import base64,io,os,sys,zipfile,json,subprocess,time
from pathlib import Path
STARTED=time.monotonic()
CAP={CAP}
PRESERVATION_RESERVE=1200
WORK=Path('/kaggle/working')
CODE=WORK/'supervised_shuffle_runner'
CODE.mkdir(exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode('{payload}'))).extractall(CODE)
os.chdir(CODE)
sys.path.insert(0,str(CODE));os.environ['PYTHONPATH']=str(CODE)
def run(*args):
    subprocess.run([sys.executable,*map(str,args)],cwd=CODE,check=True)
def attached_input(slug,owner):
    base=Path('/kaggle/input')
    candidates=[base/slug,base/'datasets'/owner/slug,base/'notebooks'/owner/slug,base/'kernels'/owner/slug]
    found=[p for p in candidates if p.is_dir()]
    if len(found)!=1:raise ValueError('Missing or ambiguous mounted input: '+slug)
    return found[0]
def remaining_stage_seconds():
    return max(0,CAP-PRESERVATION_RESERVE-(time.monotonic()-STARTED))
wheels=attached_input('img2biomass-offline-deps','meghmayur1186')
found=list(wheels.rglob('timm-1.0.30-*.whl'))
if len(found)!=1:raise ValueError('One pinned offline wheel directory required.')
run('tools/install_gpu_deps.py','--wheels',found[0].parent)
import torch
if (not torch.cuda.is_available() or 'rtx pro 6000' not in torch.cuda.get_device_name().lower()
        or torch.cuda.get_device_capability()[0]<10):
    raise RuntimeError('Actual RTX PRO6000 Blackwell required; no fallback.')
run('-m','biomass_jepa.check_gpu')
'''
    preflight=f'''import shutil,hashlib,importlib.metadata
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np,pandas as pd
from PIL import Image,ImageOps
from biomass_jepa.data import TARGETS,WEIGHTS,file_sha256,image_table
from biomass_jepa.official_heads import source_model,validate_frame
from biomass_jepa.train import write_json
from biomass_jepa.block_shuffle import permutation_seed,shuffle_square_tile
from biomass_jepa.supervised_shuffle_report import report,prepare_native_auxiliary
FOLDS=[0,1,2]
ARMS=[('control',0,0.),('shuffle_4',4,.2)]
ROOT=WORK/'supervised_one_run';ROOT.mkdir(exist_ok=True)
SOURCE_KERNEL={SOURCE!r}
SOURCE_SHA={SOURCE_SHA!r}
SOURCE_FOLDS_SHA={FOLDS_SHA!r}
SOURCE_EXTERNAL_SHA={EXTERNAL_SHA!r}
mounted=attached_input('img2biomass-grass-jepa-frozen-heads','meghmayur1186')
def unique_file(pattern):
    paths=list(mounted.rglob(pattern))
    if len(paths)!=1:raise ValueError('Missing or ambiguous pinned artifact: '+pattern)
    return paths[0]
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
    raise ValueError('Pinned source2 bytes, fold table or SSL provenance changed.')
# Strictly load the verified grass-adapted student; the trainer then unfreezes only the declared tail.
encoder,origin=source_model(adapted,'encoder')
if origin['grass_steps']!=2400 or origin['dim']!=1280:raise ValueError('Need exact completed grass-adapted student.')
del encoder
csiro=attached_input('csiro','dheerajkannaujiya')
train_csvs=list(csiro.rglob('train.csv'))
if len(train_csvs)!=1:raise ValueError('One compatible CSIRO supervised label CSV required.')
csiro=train_csvs[0].parent
frame=pd.read_csv(source_folds,dtype={{'group_id':str}})
frame['absolute_path']=[str(csiro/p) for p in frame.image_path]
if (len(frame)!=357 or frame.group_id.nunique()!=28 or frame.groupby('group_id').fold.nunique().max()!=1
        or len(frame[frame.fold.isin(FOLDS)])!=167):raise ValueError('Original cohort changed; never generate new folds.')
labels=image_table(train_csvs[0],csiro).sort_values('image_path').reset_index(drop=True)
if labels.image_path.tolist()!=frame.image_path.tolist() or not np.allclose(labels[TARGETS],frame[TARGETS],rtol=0,atol=1e-8):
    raise ValueError('Mounted labels disagree with unchanged saved folds.')
folds_csv=ROOT/'prepared/folds.csv';folds_csv.parent.mkdir(exist_ok=True)
frame.to_csv(folds_csv,index=False);validate_frame(folds_csv)
external_manifest=json.loads(external.read_text())
if len(external_manifest)!=30663 or set(frame.image_sha256)&{{r['sha256'] for r in external_manifest}}:
    raise ValueError('Original SSL provenance/exact overlap changed.')
grass=attached_input('grass-clover-datasets','pushpakbhoge')
native_csvs=list(grass.rglob('biomass_train_data.csv'))
if len(native_csvs)!=1:raise ValueError('One pinned native biomass training CSV required.')
native_raw=native_csvs[0]
if file_sha256(native_raw)!='bed4efd6f40fec31f39f5fa0f4273097e361fc7d3a3c75d51749942518fcfceb':
    raise ValueError('Native GrassClover version1 training labels changed.')
image_candidates=[p for p in grass.rglob('images') if (p/'biomass_image_train_0000.jpg').is_file()]
if len(image_candidates)!=1:raise ValueError('Missing or ambiguous native261training original scene directory.')
aux_csv=ROOT/'prepared/native_aux.csv'
excluded=set(frame.image_sha256)|{{r['sha256'] for r in external_manifest}}
aux_audit=prepare_native_auxiliary(native_raw,image_candidates[0],aux_csv,excluded,expected_rows=261)
write_json(ROOT/'prepared/native_aux_audit.json',aux_audit)
shutil.copy2(native_raw,ROOT/'prepared/native_biomass_train_data.csv')
for path in grass.rglob('biomass_test_data.csv'):
    # Preserve raw provenance separately; these heldout labels are never passed to training/evaluation.
    shutil.copy2(path,ROOT/'prepared/native_biomass_test_data_unused.csv')
for path in frame.absolute_path:
    with Image.open(path) as raw:
        if ImageOps.exif_transpose(raw).size!=(2000,1000):raise ValueError('Original full field image geometry changed.')
with Image.open(frame.iloc[0].absolute_path) as raw:
    tile=ImageOps.exif_transpose(raw).convert('RGB').crop((0,0,1000,1000))
    seed=permutation_seed(frame.iloc[0].image_sha256,4,0,0)
    shuffled=shuffle_square_tile(tile,4,seed)
    if shuffled.tobytes()!=shuffle_square_tile(tile,4,seed).tobytes() or shuffled.histogram()!=tile.histogram():
        raise ValueError('Permutation must conserve all pixels deterministically.')
# Delta checkpoints contain only trainable encoder tail/head, their optimizer and SWA states.
if shutil.disk_usage(WORK).free<8*1024**3:raise ValueError('Need8GiB for six paired delta/resume checkpoints and reports.')
shutil.copy2(source_folds,ROOT/'prepared/source_folds.csv')
shutil.copy2(ssl_log,ROOT/'historical_ssl_metrics.jsonl')
shutil.copy2(CODE/'paper/supervised_shuffle_protocol.md',ROOT/'ANALYSIS_PLAN.md')
provenance={{'checked_ist':datetime.now(ZoneInfo('Asia/Calcutta')).isoformat(),
    'experiment':'supervised_encoder_tail_online_shuffle_v1','source_kernel':SOURCE_KERNEL,
    'source_sha256':SOURCE_SHA,'source_origin':origin,'source_progress':progress,
    'source_fold_sha256':SOURCE_FOLDS_SHA,'mapped_fold_sha256':file_sha256(folds_csv),
    'source_external_manifest_sha256':SOURCE_EXTERNAL_SHA,'exact_ssl_supervised_overlap':0,
    'near_duplicate_overlap':'Not ruled out; exact byte SHA audit only.',
    'code_base_git_sha':{revision!r},'bundle_files_sha256':{{str(p.relative_to(CODE)):file_sha256(p) for p in CODE.rglob('*.py')}},
    'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'cuda':torch.version.cuda,
    'software':{{n:importlib.metadata.version(n) for n in ['torchao','timm','torchvision','numpy','pandas','scikit-learn','Pillow','matplotlib']}},
    'folds':FOLDS,'images':357,'evaluated_images':167,'complete_oof':False,'competition_score':False,
    'seed':42,'shuffle_probability':.2,'epochs':200,'swa_last_epochs':40,
    'source_weight_updates':True,'unfreeze_last':2,'encoder_final_norm_trainable':True,
    'predictor_used':False,'new_ssl':False,'augmentation':'online varying epoch/image/fold SHA; quarter-turns only',
    'validation_shuffle':False,'supervised_labels':'CSIRO357 plus261native GrassClover dry grass/clover/weeds/total through a separate auxiliary head',
    'auxiliary':aux_audit,'aux_weight':.25,'aux_shuffle':False,
    'distinct_training_photos_by_fold':{{str(f):int((frame.fold!=f).sum())+261 for f in FOLDS}},
    'limits':'Reused development dates are site proxies; one seed; no protected confirmation or preregistration.'}}
write_json(ROOT/'PROVENANCE.json',provenance)
print('SUPERVISED PREFLIGHT VERIFIED:',json.dumps(provenance),flush=True)
'''
    experiment='''states=[]
# Only completed200-epoch stages are compared. Each fold is paired before progressing.
# The first stage measures throughput inside the one authorized execution; no pilot GPU job.
try:
    for fold in FOLDS:
        for name,grid,probability in ARMS:
            remaining=remaining_stage_seconds()
            if remaining<600:raise RuntimeError('Preservation boundary before next stage; keep existing paired outputs.')
            previous=[s for s in states if s['status']=='completed_fixed_endpoint']
            if previous:
                estimate=max(s['elapsed_seconds'] for s in previous)*1.25
                stages_needed=2 if name=='control' else 1
                if remaining<estimate*stages_needed:
                    raise RuntimeError('Insufficient measured budget for next matched fold stage; no shorter endpoint.')
            stage_seconds=remaining/2 if name=='control' else remaining
            started=time.monotonic()
            run('-m','biomass_jepa.supervised_shuffle',
                '--init',adapted,'--folds-csv',folds_csv,'--aux-folds-csv',aux_csv,'--aux-weight','.25','--output',ROOT/'arms'/name,
                '--precision','fp8','--folds',*FOLDS,'--run-fold',fold,
                '--shuffle-grid',grid,'--shuffle-prob',probability,
                '--unfreeze-last','2','--encoder-lr','5e-6','--head-lr','1e-4',
                '--batch-size','16','--weight-decay','.01','--workers','2',
                '--epochs','200','--swa-last-epochs','40','--seed','42',
                '--max-hours',stage_seconds/3600)
            path=ROOT/'arms'/name/f'fold_{fold}'/'metrics.json'
            metrics=json.loads(path.read_text()) if path.is_file() else {}
            complete=metrics.get('status')=='completed_fixed_endpoint' and metrics.get('completed_epochs')==200
            states.append({'arm':name,'fold':fold,'status':'completed_fixed_endpoint' if complete else 'incomplete',
                'elapsed_seconds':time.monotonic()-started})
            write_json(ROOT/'ARM_STATUS.json',{'stages':states,'elapsed_seconds':time.monotonic()-STARTED})
            if not complete:raise RuntimeError('Stage incomplete; paired fixed endpoints retained separately.')
except Exception as exc:
    write_json(ROOT/'EXECUTION_BOUNDARY.json',{'type':type(exc).__name__,'message':str(exc),
        'preserve_existing_artifacts':True,'no_automatic_retry':True})
    print('SUPERVISED RUN STOPPED AT BOUNDARY:',type(exc).__name__,flush=True)
finally:
    write_json(ROOT/'ARM_STATUS.json',{'stages':states,'elapsed_seconds':time.monotonic()-STARTED})
    # Preserve inventory before plotting/reporting; analysis failure cannot hide checkpoint evidence.
    inventory={str(p.relative_to(ROOT)):{'sha256':file_sha256(p),'bytes':p.stat().st_size}
        for p in ROOT.rglob('*') if p.is_file() and p.name!='CHECKPOINT_INVENTORY.json'
        and (p.suffix=='.pt' or p.name=='PROVENANCE.json')}
    write_json(ROOT/'CHECKPOINT_INVENTORY.json',inventory)
    try:
        result=report(ROOT,folds_csv,draws=10000,seed=42,ssl_log=ROOT/'historical_ssl_metrics.jsonl',examples=True)
        print('SUPERVISED REPORT:',json.dumps(result),flush=True)
    except Exception as exc:
        write_json(ROOT/'REPORT_FAILURE.json',{'type':type(exc).__name__,'message':str(exc),
            'weights_preserved':True,'report_not_verified':True})
        print('REPORT FAILED; checkpoint inventory preserved:',type(exc).__name__,flush=True)
'''
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
        'cells':[cell('# Supervised grass-I-JEPA encoder learning with online spatial permutation\nTwo matched arms: control and4×4p0.2. Last2encoderblocks+finalnorm and gated head train together.\nFolds0/1/2, one seed;167/357 reused CSI development photos;261native auxiliary training scenes with separate labels. No new SSL, predictor or DINO.\n',False),
                 cell(bootstrap),cell(preflight),cell(experiment),
                 cell('## Preserve outputs\nRetain supervised_one_run, source hashes and all delta selected/latest/SWA states.\nInteractive execution: Quick Save with Always save output and verify saved Output files.\nThis experiment does not assert successful GitHub weight backup or journal acceptance.\n',False)]}
    meta={'id':KERNEL,'title':'Img2BioMass JEPA Supervised Shuffle','code_file':'supervised_shuffle_experiment.ipynb',
        'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_tpu':False,
        'enable_internet':False,'machine_shape':'NvidiaRtxPro6000',
        'dataset_sources':['dheerajkannaujiya/csiro','meghmayur1186/img2biomass-offline-deps','pushpakbhoge/grass-clover-datasets/1'],
        'competition_sources':['arc-prize-2026-arc-agi-3'],'kernel_sources':[SOURCE]}
    return notebook,meta


def prepare(root):
    root=Path(root);out=root/'outputs/supervised_shuffle_launch';out.mkdir(parents=True,exist_ok=True)
    if (out/'launch.json').exists():raise ValueError('Never rebuild a dispatched notebook.')
    notebook,meta=build(root)
    (out/meta['code_file']).write_text(json.dumps(notebook,indent=1))
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    return out


if __name__=='__main__':
    print(prepare(Path(__file__).resolve().parents[1]))
