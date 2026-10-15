"""Fixed validation-image precision comparison; no fitting or hidden-test labels."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import BiomassImages, TARGETS, file_sha256
from .infer import reconstruct_model
from .metrics import scores
from .models import configure_fp8
from .train import load_checkpoint, write_json


def compare_predictions(truth, predictions):
    reference = predictions['fp8_bf16']
    result = {}
    reference_score = scores(truth, reference)['global_weighted_r2_raw']
    for name, values in predictions.items():
        if values.shape != truth.shape or not np.isfinite(values).all():
            raise ValueError('Invalid precision-parity predictions.')
        difference = np.abs(values-reference)
        score = scores(truth, values)['global_weighted_r2_raw']
        result[name] = {'raw_global_weighted_r2':score,
            'score_delta_vs_fp8_bf16':None if score is None or reference_score is None else score-reference_score,
            'per_target_max_absolute_delta_g':dict(zip(TARGETS, difference.max(0).tolist())),
            'per_target_median_absolute_delta_g':dict(zip(TARGETS, np.median(difference, axis=0).tolist()))}
    return result


@torch.inference_mode()
def run(weights, folds_csv, image_root, output, fold=0, images=20, initial_weights=None):
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA GPU required; CPU checks do not establish precision parity.')
    if images < 2:
        raise ValueError('Need at least two declared validation images.')
    weights, output = Path(weights), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((weights/'inference_manifest.json').read_text())
    if manifest['source_kernel'] != 'meghmayur1186/capstone' or manifest['source_version'] != 9 or fold not in manifest['folds']:
        raise ValueError('This diagnostic is declared for retained selected v9 weights.')
    initial_path = Path(initial_weights) if initial_weights else weights/'jepa_encoder.pt'
    for filename, path in [('jepa_encoder.pt',initial_path),(f'fold_{fold}_best.pt',weights/f'fold_{fold}_best.pt')]:
        if file_sha256(path) != manifest['sha256'][filename]:
            raise ValueError('Selected parity weights changed.')
    selected = load_checkpoint(weights/f'fold_{fold}_best.pt')
    sig = selected['signature']
    if sig != manifest['signature'] or sig['init_sha256'] != manifest['sha256']['jepa_encoder.pt']:
        raise ValueError('Checkpoint initialization/signature mismatch.')
    frame = pd.read_csv(folds_csv, dtype={'group_id':str})
    frame = frame[frame.fold == fold].sort_values('image_path').head(images).reset_index(drop=True)
    if len(frame) < images:
        raise ValueError('Not enough validation images for the declared diagnostic.')
    frame['absolute_path'] = [str(Path(image_root)/p) for p in frame.image_path]
    for path, digest in zip(frame.absolute_path, frame.image_sha256):
        if file_sha256(path) != digest:
            raise ValueError('Validation image bytes differ from the prepared data.')
    batches = DataLoader(BiomassImages(frame, sig['size'], views=sig['views']),
                         batch_size=2, num_workers=0, shuffle=False)
    initial = load_checkpoint(initial_path)
    predictions = {}
    for mode, dtype in [('fp8_bf16', torch.bfloat16), ('bf16', torch.bfloat16), ('fp16', torch.float16)]:
        model = reconstruct_model(initial, selected).cuda().eval()
        if mode == 'fp8_bf16':
            configure_fp8(model, 'fp8', torch.device('cuda'))
        # Conversion must preserve all saved master weights/buffers used by inference.
        state = model.state_dict()
        for name, value in selected['model'].items():
            torch.testing.assert_close(state[name].cpu(), value, rtol=0, atol=0)
        values = []
        for tiles, _, _ in tqdm(batches, desc='Precision parity '+mode, mininterval=2):
            with torch.autocast('cuda', dtype=dtype):
                values.append(model(tiles.cuda()).float().cpu().numpy())
        predictions[mode] = np.concatenate(values)
        print('PRECISION PARITY completed:',mode,flush=True)
        del state, model
        gc.collect(); torch.cuda.empty_cache()
    truth = frame[TARGETS].to_numpy()
    record = {'status':'completed_actual_gpu_parity','device':torch.cuda.get_device_name(),
              'source_version':9,'fold':fold,'images':len(frame),
              'image_selection':f'first{images} sorted fold{fold} validation images, declared before execution',
              'test_labels_accessed':False,'results':compare_predictions(truth, predictions),
              'limit':'Same RTX device; this does not establish cross-device T4 equivalence.'}
    table = frame[['image_path','group_id','fold']+TARGETS].copy()
    for mode, values in predictions.items():
        for index, target in enumerate(TARGETS):table[mode+'_'+target] = values[:,index]
    table.to_csv(output/'predictions.csv',index=False)
    write_json(output/'parity.json',record)
    print('PRECISION PARITY RESULT:',json.dumps(record),flush=True)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['weights','folds-csv','image-root','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--fold',type=int,default=0);p.add_argument('--images',type=int,default=20)
    p.add_argument('--initial-weights',help='Shared encoder export may be a separate Kaggle input; must match its manifest hash.')
    run(**vars(p.parse_args()))


if __name__ == '__main__':main()
