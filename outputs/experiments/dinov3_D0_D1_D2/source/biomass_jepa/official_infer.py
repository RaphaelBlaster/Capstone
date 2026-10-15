"""Offline inference with the original official source plus selected changing weights."""
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

from .data import TARGETS, BiomassImages, file_sha256
from .infer import submission_table
from .official_ijepa import (OfficialEncoder, OfficialBiomassModel, make_predictor,
                            strip_ddp, OFFICIAL_MODEL, OFFICIAL_REVISION)
from .train import load_changing_state


def validate_manifest_origin(manifest):
    version = manifest.get('source_version')
    if (manifest.get('package_format') != 'official_ijepa_transfer_v1'
            or manifest.get('source_kernel') != 'meghmayur1186/capstone'
            or type(version) is not int or version < 18
            or manifest.get('folds') != [0,1]
            or manifest.get('signature',{}).get('model') != OFFICIAL_MODEL):
        raise ValueError('Need the explicit official I-JEPA two-fold retained package.')
    if version != 18 and (manifest.get('runner_source_version') != 18
            or manifest.get('source_capture') not in {'interactive_quick_save','saved_run'}
            or manifest.get('saved_runner_archive_sha256') != manifest.get('original_runner_archive_sha256')
            or not manifest.get('saved_runner_archive_sha256')):
        raise ValueError('Quick-saved outputs must retain verified original runner identity.')


def reconstruct_model(original, selected):
    signature = selected['signature']
    if (signature.get('model') != OFFICIAL_MODEL
            or signature.get('encoder_initialization') != 'official_ijepa'
            or signature.get('official_repository_revision') != OFFICIAL_REVISION
            or signature.get('official_source_epoch') != original.get('epoch')
            or signature.get('last_blocks') != (2 if signature.get('tiny') else 8)
            or signature.get('predictor_readout') != 'masked_latent'
            or signature.get('predictor_dropout') != 0
            or signature.get('head') != 'predictor'
            or selected.get('checkpoint_format') != 'changing_parameters_v1'):
        raise ValueError('Selected official model has incompatible source, scope or predictor provenance.')
    expected = (32,2) if signature.get('tiny') else (384,12)
    if (signature['predictor_width'],signature['predictor_depth']) != expected:
        raise ValueError('Official predictor dimensions differ from the paired checkpoint.')
    encoder = OfficialEncoder(size=signature['size'],last_blocks=signature['last_blocks'],tiny=signature['tiny'])
    encoder.backbone.load_state_dict(strip_ddp(original['encoder']),strict=True)
    predictor = make_predictor(encoder)
    predictor.load_state_dict(strip_ddp(original['predictor']),strict=True)
    model = OfficialBiomassModel(encoder,predictor,hidden=signature['predictor_width'],
        auxiliary_bins=signature.get('auxiliary_bins',0),dropout=signature.get('head_dropout',.1),
        output_space=signature['output_space'])
    # All grass-adapted parameters are also downstream-trainable and present in
    # the selected export. Frozen blocks/positions retain the original source.
    # The teacher is only needed during SSL, so it is never instantiated here.
    load_changing_state(model,selected['model'])
    return model


@torch.inference_mode()
def infer(weights, source_checkpoint, test_csv, image_root, output, batch_size=2,
          workers=2, require_cuda=False, precision='fp16'):
    started = time.monotonic()
    weights, source_checkpoint = Path(weights), Path(source_checkpoint)
    manifest = json.loads((weights/'inference_manifest.json').read_text())
    signature = manifest['signature']
    validate_manifest_origin(manifest)
    required = {f'fold_{fold}_best.pt' for fold in [0,1]}
    if not required <= set(manifest.get('sha256',{})):
        raise ValueError('Official manifest must hash both selected models.')
    if file_sha256(source_checkpoint) != manifest['official_source_sha256']:
        raise ValueError('Original official source input changed.')
    for relative,digest in manifest['sha256'].items():
        if Path(relative).name != relative or file_sha256(weights/relative) != digest:
            raise ValueError('Selected official artifact path or bytes differ.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if require_cuda and (device.type != 'cuda' or 'T4' not in torch.cuda.get_device_name()):
        raise RuntimeError('The competition inference notebook requires its T4 GPU.')
    if precision == 'fp16' and device.type != 'cuda':
        raise RuntimeError('CPU functional inference must explicitly request FP32.')
    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    print('OFFICIAL INFERENCE DEVICE:',torch.cuda.get_device_name() if device.type=='cuda'
          else 'CPU functional validation only',flush=True)
    test = pd.read_csv(test_csv)
    paths = test.image_path.drop_duplicates().tolist()
    frame = pd.DataFrame({'image_path':paths,'absolute_path':[str(Path(image_root)/p) for p in paths]})
    for target in TARGETS: frame[target] = 0.
    batches = DataLoader(BiomassImages(frame,signature['size'],views=signature['views']),
        batch_size=batch_size,num_workers=workers,shuffle=False,pin_memory=device.type=='cuda')
    original = torch.load(source_checkpoint,map_location='cpu',weights_only=True,mmap=True)
    total = np.zeros((len(paths),5),dtype=np.float64)
    for fold in [0,1]:
        selected = torch.load(weights/f'fold_{fold}_best.pt',map_location='cpu',weights_only=True)
        if selected['signature'] != signature:
            raise ValueError('Selected official models belong to different labeled recipes.')
        model = reconstruct_model(original,selected).to(device).eval()
        for tiles,_,indices in tqdm(batches,desc=f'Official I-JEPA inference fold {fold}',mininterval=2):
            with torch.autocast('cuda',dtype=torch.float16) if precision=='fp16' else contextlib.nullcontext():
                predictions = model(tiles.to(device,non_blocking=True)).float().cpu().numpy()
            total[indices.numpy()] += predictions
        del model,selected
        gc.collect()
        if device.type=='cuda': torch.cuda.empty_cache()
    result = submission_table(test,paths,total/2)
    output = Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(output,index=False)
    evidence = {'source_kernel':manifest['source_kernel'],'source_version':manifest['source_version'],
        'arm':manifest['arm'],'folds':[0,1],'images':len(paths),'rows':len(result),
        'seconds':time.monotonic()-started,'precision':precision,'tta':False,
        'ensemble':'equal arithmetic mean in grams','model':OFFICIAL_MODEL,
        'grass_domain_steps':signature['grass_domain_steps'],'test_labels_accessed':False,
        'official_source_sha256':manifest['official_source_sha256'],
        'weights_manifest_sha256':file_sha256(weights/'inference_manifest.json'),
        'submission_sha256':file_sha256(output)}
    output.with_suffix('.inference.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps(evidence),flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',required=True)
    parser.add_argument('--source-checkpoint',required=True)
    parser.add_argument('--test-csv',required=True)
    parser.add_argument('--image-root',required=True)
    parser.add_argument('--output',default='submission.csv')
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--require-cuda',action='store_true')
    parser.add_argument('--precision',choices=['fp16','fp32'],default='fp16')
    infer(**vars(parser.parse_args()))
