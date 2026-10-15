"""Offline saved-head inference and held-out FP16 validation; never optimize weights."""
import argparse
import contextlib
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import TARGETS,BiomassImages,file_sha256
from .infer import submission_table
from .metrics import scores
from .official_heads import FrozenHeadInference


def validate_package(folder):
    folder=Path(folder)
    manifest=json.loads((folder/'inference_manifest.json').read_text())
    required={'fold_0_selected.pt','fold_1_selected.pt','folds.csv'}
    if (manifest.get('package_format')!='official_frozen_heads_v1'
            or manifest.get('folds')!=[0,1] or not required<=set(manifest.get('sha256',{}))
            or manifest.get('arm') not in {'foundation_encoder','grass_encoder','grass_predictor'}):
        raise ValueError('Explicit two-fold frozen-head package required.')
    for name,digest in manifest['sha256'].items():
        if Path(name).name!=name or file_sha256(folder/name)!=digest:
            raise ValueError('Package artifact bytes/path differ.')
    return manifest


@torch.inference_mode()
def predict_frame(model,frame,size,device,precision,batch_size,workers):
    result=np.zeros((len(frame),5),dtype=np.float64)
    loader=DataLoader(BiomassImages(frame,size,train=False,views=2),batch_size=batch_size,
                      num_workers=workers,shuffle=False,pin_memory=device.type=='cuda')
    for tiles,_,indices in tqdm(loader,desc='Frozen-head inference',mininterval=2.):
        with (torch.autocast('cuda',dtype=torch.float16) if precision=='fp16' else contextlib.nullcontext()):
            predictions=model(tiles.to(device)).float().cpu().numpy()
        if not np.isfinite(predictions).all() or (predictions<0).any():
            raise FloatingPointError('Nonfinite/negative biomass inference.')
        result[indices.numpy()]=predictions
    return result


def infer(weights,source_checkpoint,test_csv,image_root,output='submission.csv',
          precision='fp16',batch_size=2,workers=2,require_cuda=False,validate_labeled=True):
    started=time.monotonic();weights=Path(weights);manifest=validate_package(weights)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if require_cuda and (device.type!='cuda' or 'T4' not in torch.cuda.get_device_name()):
        raise RuntimeError('Actual T4 inference GPU required.')
    if precision=='fp16' and device.type!='cuda':raise RuntimeError('CPU functional inference requires explicitFP32.')
    if file_sha256(source_checkpoint)!=manifest['frozen_source_sha256']:
        raise ValueError('Frozen source differs from the exact feature checkpoint.')
    frame=pd.read_csv(weights/'folds.csv',dtype={'group_id':str})
    if frame.groupby('group_id').fold.nunique().max()!=1 or frame.image_path.duplicated().any():
        raise ValueError('Fold/image mapping changed.')
    test=pd.read_csv(test_csv)
    if 'target' in test:raise ValueError('Test labels must not be supplied.')
    paths=test.image_path.drop_duplicates().tolist()
    test_frame=pd.DataFrame({'image_path':paths,'absolute_path':[str(Path(image_root)/p) for p in paths]})
    for target in TARGETS:test_frame[target]=0.
    total=np.zeros((len(paths),5),dtype=np.float64);validation=[]
    for fold in (0,1):
        checkpoint=torch.load(weights/f'fold_{fold}_selected.pt',weights_only=True,map_location='cpu')
        signature=checkpoint['signature'];origin=checkpoint['feature_origin']
        if (signature!=manifest['head_signature'] or origin!=manifest['feature_origin']
                or signature['selection']!='fixed_final_swa_endpoint_not_best_validation_epoch'
                or checkpoint['epochs']!=signature['epochs']):
            raise ValueError('Selected head architecture, training endpoint or feature origin changed.')
        if not origin['tiny'] and signature['precision']!='fp8':
            raise ValueError('Only actual FP8-trained real heads may be submitted.')
        model=FrozenHeadInference(source_checkpoint,checkpoint).to(device).eval()
        total+=predict_frame(model,test_frame,origin['size'],device,precision,batch_size,workers)
        if validate_labeled:
            held=frame[frame.fold==fold].copy().reset_index(drop=True)
            held['absolute_path']=[str(Path(image_root)/p) for p in held.image_path]
            for path,digest in zip(held.absolute_path,held.image_sha256):
                if file_sha256(path)!=digest:raise ValueError('Validation image bytes differ from training cohort.')
            prediction=predict_frame(model,held,origin['size'],device,precision,batch_size,workers)
            for j,name in enumerate(TARGETS):held['pred_'+name]=prediction[:,j]
            validation.append(held[['image_path','group_id','fold']+TARGETS+['pred_'+t for t in TARGETS]])
        del model,checkpoint;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    submission=submission_table(test,paths,total/2)
    submission.to_csv(output,index=False)
    parity=None
    if validation:
        validation=pd.concat(validation).sort_values('image_path')
        validation.to_csv(output.with_suffix('.validation.csv'),index=False)
        parity=scores(validation[TARGETS].to_numpy(),validation[['pred_'+t for t in TARGETS]].to_numpy())
    evidence={'source_kernel':manifest['source_kernel'],'source_version':manifest['source_version'],
        'arm':manifest['arm'],'folds':[0,1],'precision':precision,'tta':False,'test_labels_accessed':False,
        'actual_gpu':torch.cuda.get_device_name() if device.type=='cuda' else 'CPU functional test',
        'images':len(paths),'rows':len(submission),'seconds':time.monotonic()-started,
        'frozen_source_sha256':manifest['frozen_source_sha256'],
        'weights_manifest_sha256':file_sha256(weights/'inference_manifest.json'),
        'submission_sha256':file_sha256(output),'validation_scores':parity,
        'validation_role':'Same fixed heads on held-out labeled images, no fitting or calibration.'}
    output.with_suffix('.inference.json').write_text(json.dumps(evidence,indent=2))
    print('FROZEN HEAD INFERENCE RESULT:',json.dumps(evidence),flush=True)
    return submission


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('weights','source-checkpoint','test-csv','image-root'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--output',default='submission.csv')
    parser.add_argument('--precision',choices=['fp16','fp32'],default='fp16')
    parser.add_argument('--batch-size',type=int,default=2);parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--require-cuda',action='store_true')
    parser.add_argument('--skip-labeled-validation',action='store_true',help='Hidden rerun: no fitting; visible run already measured parity.')
    args=vars(parser.parse_args());args['validate_labeled']=not args.pop('skip_labeled_validation')
    infer(**args)
