"""Frozen official I-JEPA representations and resumable, augmented biomass heads.

No SSL or encoder updates occur here. Real head optimization requires RTX PRO6000
and eligible FP8 linears; --tiny/--precision bf16 is for CPU functional tests only.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from . import train as common
from .data import TARGETS, WEIGHTS, BiomassImages, file_sha256
from .metrics import scores, group_bootstrap
from .models import configure_fp8
from .official_ijepa import (OFFICIAL_MODEL, OFFICIAL_REVISION, OfficialEncoder,
                            OfficialBiomassModel, make_predictor, strip_ddp)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def feature_code_revision():
    root=Path(__file__).parent
    return digest_json({str(path.relative_to(root)):file_sha256(path)
        for path in sorted(root.rglob('*')) if path.is_file()
        and (path.suffix=='.py' or path.name in {'LICENSE','SOURCE.json'})})


def require_runtime(args):
    if not args.tiny and (args.precision != 'fp8' or not torch.cuda.is_available()
            or 'rtx pro 6000' not in torch.cuda.get_device_name().lower()
            or torch.cuda.get_device_capability()[0] < 10):
        raise RuntimeError('Actual RTX PRO6000 Blackwell/FP8 required; no training fallback.')
    return common.device_for(args)


def validate_frame(path):
    frame = pd.read_csv(path, dtype={'group_id': str})
    if (frame.image_path.duplicated().any() or frame.image_sha256.duplicated().any()
            or frame.groupby('group_id').fold.nunique().max() != 1):
        raise ValueError('Duplicate image or acquisition group crossing folds.')
    for path, digest in zip(frame.absolute_path, frame.image_sha256):
        if file_sha256(path) != digest:
            raise ValueError('Prepared image bytes changed.')
    return frame


def quarter_turn(tiles, index):
    """Only 0/90/180/270 degrees: exact pixel permutations, no reflection/crop."""
    if not 0 <= index < 4:
        raise ValueError('Quarter-turn index outside 0..3.')
    return torch.rot90(tiles, index, (-2, -1)).contiguous()


def source_model(path, representation, tiny=False):
    saved = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
    source_hash=file_sha256(path)
    # The original control can start directly from the already mounted official
    # checkpoint; it does not need to wait for a new grass adaptation/export.
    if 'official_source' in saved:
        if (saved['official_source'].get('repository_revision')!=OFFICIAL_REVISION
                or source_hash!='2dbfad65ef9fc869ff9bcba6222dfc65efa580de17365cde100a83138f1038cd'
                or tiny):
            raise ValueError('Unexpected original official compact source.')
        encoder=OfficialEncoder(224,0,False)
        encoder.backbone.load_state_dict(strip_ddp(saved['encoder']),strict=True)
        if representation=='encoder':model=encoder
        elif representation=='predictor':
            predictor=make_predictor(encoder)
            predictor.load_state_dict(strip_ddp(saved['predictor']),strict=True)
            model=OfficialBiomassModel(encoder,predictor,auxiliary_bins=0)
        else:raise ValueError('Unknown frozen representation.')
        model.requires_grad_(False);model.eval()
        return model,{'source_sha256':source_hash,'size':224,'dim':encoder.dim,
            'representation':representation,'tiny':False,'model':OFFICIAL_MODEL,
            'revision':OFFICIAL_REVISION,'grass_steps':0,'original_compact_sha256':source_hash}
    signature = saved['signature']
    if (saved.get('initialization_kind') != 'official_ijepa_domain_v1'
            or saved.get('official_repository_revision') != OFFICIAL_REVISION
            or signature.get('model') != OFFICIAL_MODEL or signature.get('tiny') != tiny
            or saved.get('official_source_epoch', 0) < 1
            or saved.get('domain_adaptation_status') not in {'original_official_weights', 'completed_steps'}):
        raise ValueError('Need a complete original or adapted official source export.')
    encoder = OfficialEncoder(signature['size'], 0, tiny)
    encoder.load_state_dict(saved['encoder'], strict=True)
    if representation == 'encoder':
        model = encoder
    elif representation == 'predictor':
        predictor = make_predictor(encoder)
        predictor.load_state_dict(saved['predictor'], strict=True)
        model = OfficialBiomassModel(encoder, predictor, auxiliary_bins=0)
    else:
        raise ValueError('Unknown frozen representation.')
    model.requires_grad_(False)
    model.eval()
    metadata = {'source_sha256': source_hash, 'size': signature['size'],
        'dim': encoder.dim, 'representation': representation, 'tiny': tiny,
        'model': OFFICIAL_MODEL, 'revision': OFFICIAL_REVISION,
        'grass_steps': saved['step'], 'original_compact_sha256': saved['official_compact_sha256']}
    return model, metadata


def cache_features(args):
    device = require_runtime(args)
    frame = validate_frame(args.folds_csv)
    model, origin = source_model(args.init, args.representation, args.tiny)
    model.to(device)
    precision = configure_fp8(model, args.precision, device)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    grid = getattr(args, 'shuffle_grid', 0)
    if grid not in (0, 4, 10) or (grid and (args.representation != 'encoder'
            or (not args.tiny and origin['grass_steps'] != 2400))):
        raise ValueError('Shuffle cache requires the completed adapted encoder and grid 4/10.')
    signature = {**origin, 'precision': args.precision, 'views': 2, 'batch_size':args.batch_size,
        'images': frame[['image_path','image_sha256']].to_dict('records'),
        'augmentation': 'four_complete_tile_quarter_turns_identity_validation',
        'shuffle_grid': grid, 'shuffle_version': 'blockshuffle-v1', 'orientations': [0,1,2,3],
        'preprocessing': 'exif_rgb_two_full_square_halves_shuffle_before_bilinear_resize_imagenet_normalize_rot90_after',
        'patch_size': 14, 'shape': [len(frame),2,origin['dim']],
        'code_revision': feature_code_revision()}
    manifest_path=output/'manifest.json'
    if manifest_path.exists():
        completed=json.loads(manifest_path.read_text())
        if completed['status']!='complete_features' or completed['signature']!=signature:
            raise ValueError('Completed feature cache configuration differs.')
        # Hashes are immutable commitments, not values to silently recompute and adopt.
        load_feature_bank(output,frame,args,torch.device('cpu'))
        return
    started = time.monotonic()
    for orientation in range(4):
        path = output/f'orientation_{orientation}.pt'
        saved = torch.load(path, weights_only=True) if path.exists() else None
        if saved and (saved['signature'] != signature or saved.get('orientation') != orientation):
            raise ValueError('Feature cache belongs to another image/source/precision configuration.')
        if saved and (list(saved['features'].shape) != signature['shape']
                or not torch.isfinite(saved['features']).all()
                or not 0 <= saved['next_image'] <= len(frame)):
            raise ValueError('Invalid feature cache tensor or cursor.')
        if saved and saved['next_image'] == len(frame):
            continue
        features = saved['features'] if saved else torch.zeros(len(frame),2,origin['dim'])
        cursor = saved['next_image'] if saved else 0
        batches = DataLoader(BiomassImages(frame, origin['size'], train=False, views=2,
            block_shuffle_grid=grid, block_shuffle_variant=orientation),
            batch_size=args.batch_size, num_workers=args.workers)
        model.eval()
        for tiles, _, indices in tqdm(batches, desc=f'Frozen {args.representation} rotation {orientation+1}/4', mininterval=2.):
            if int(indices[-1]) < cursor:
                continue
            with torch.no_grad(), common.amp(device):
                images = quarter_turn(tiles, orientation).to(device).flatten(0,1)
                tokens = (model(images) if args.representation == 'encoder'
                          else model.patch_predictions(images))
                value = tokens.mean(1).reshape(len(tiles),2,origin['dim']).float().cpu()
            if not torch.isfinite(value).all():
                raise FloatingPointError('Nonfinite frozen features.')
            features[indices] = value
            cursor = int(indices[-1])+1
            # Small cache writes only; interrupted extraction replays at most one batch.
            common.atomic_save({'signature': signature, 'orientation': orientation,
                'features': features, 'next_image': cursor}, path)
            if common.STOP_REQUESTED or time.monotonic()-started > args.max_hours*3600-60:
                return
    hashes = {f'orientation_{i}.pt': file_sha256(output/f'orientation_{i}.pt') for i in range(4)}
    common.write_json(output/'manifest.json', {'status':'complete_features', 'signature':signature,
        'sha256':hashes,'runtime':{'gpu':torch.cuda.get_device_name() if device.type=='cuda' else 'CPU tiny test',
        'precision':precision}, 'seconds':time.monotonic()-started})


class GatedRegressionHead(nn.Module):
    """Shared small readout for observed and predicted patch means; no GDConv."""
    def __init__(self, dim, scale, median, width=512, dropout=.2):
        super().__init__()
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32).clone())
        self.projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim,width), nn.GELU())
        self.gate_tanh = nn.Linear(width,128)
        self.gate_sigmoid = nn.Linear(width,128)
        self.gate_output = nn.Linear(128,1)
        self.readout = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,256),nn.GELU(),
                                     nn.Dropout(dropout),nn.Linear(256,3))
        value = (torch.as_tensor(median,dtype=torch.float32)/self.scale).clamp_min(1e-4)
        with torch.no_grad():
            self.readout[-1].weight.zero_()
            self.readout[-1].bias.copy_(value+torch.log(-torch.expm1(-value)))

    def forward(self, features):
        projected = self.projection(features)
        logits = self.gate_output(torch.tanh(self.gate_tanh(projected))*torch.sigmoid(self.gate_sigmoid(projected)))
        weight = logits.float().softmax(1).to(projected.dtype)
        pooled = (weight*projected).sum(1)
        primitive = F.softplus(self.readout(pooled).float())*self.scale
        green,dead,clover = primitive.unbind(1)
        return torch.stack((green,dead,clover,green+clover,green+clover+dead),1)


def load_feature_bank(cache, frame, args, device):
    cache = Path(cache)
    manifest = json.loads((cache/'manifest.json').read_text())
    signature = manifest['signature']
    if (manifest['status'] != 'complete_features'
            or signature['images'] != frame[['image_path','image_sha256']].to_dict('records')
            or signature['tiny'] != args.tiny or signature['precision'] != args.precision):
        raise ValueError('Complete feature cache matching this image cohort/precision required.')
    if signature.get('code_revision') and signature['code_revision'] != feature_code_revision():
        raise ValueError('Feature cache code revision differs from the running code.')
    names = [f'orientation_{i}.pt' for i in range(4)]
    if set(manifest['sha256']) != set(names):
        raise ValueError('Exactly four quarter-turn orientations required.')
    features = []
    for i, name in enumerate(names):
        path = cache/name
        if file_sha256(path) != manifest['sha256'][name]:
            raise ValueError('Feature cache bytes changed.')
        value = torch.load(path, weights_only=True)
        if (value['signature'] != signature or value['next_image'] != len(frame)
                or value.get('orientation',i) != i):
            raise ValueError('Incomplete or mixed-source feature cache.')
        feature = value['features']
        if (feature.shape != (len(frame),2,signature['dim']) or feature.dtype != torch.float32
                or not torch.isfinite(feature).all()):
            raise ValueError('Feature shape, dtype or finite-value check failed.')
        features.append(feature)
    return torch.stack(features).to(device), manifest


def select_training_features(control, shuffled, orientation, indices, probability, generator):
    """One label per original image; private RNG leaves dropout/order RNG unchanged."""
    chosen = control[orientation, indices]
    if shuffled is not None:
        use = torch.rand(len(indices), generator=generator, device=control.device) < probability
        chosen = torch.where(use[:,None,None], shuffled[orientation,indices], chosen)
    return chosen


def model_state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def fit_heads(args):
    device = require_runtime(args); frame = validate_frame(args.folds_csv)
    if args.folds not in ([0,1], [0,1,2]):
        raise ValueError('Exactly two or three original folds required.')
    features, manifest = load_feature_bank(args.cache, frame, args, device)
    if manifest['signature'].get('shuffle_grid',0) != 0:
        raise ValueError('Validation requires an unshuffled control cache.')
    shuffle_cache = getattr(args,'shuffle_cache',None)
    probability = getattr(args,'shuffle_prob',.35)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('Shuffle probability must be in [0,1].')
    shuffled = None
    if shuffle_cache:
        shuffled, shuffle_manifest = load_feature_bank(shuffle_cache, frame, args, device)
        left = dict(manifest['signature']);right = dict(shuffle_manifest['signature'])
        grid = right.pop('shuffle_grid',None);left.pop('shuffle_grid',None)
        if grid not in (4,10) or left != right or left['representation'] != 'encoder':
            raise ValueError('A paired shuffle cache must match the exact control source and preprocessing.')
    targets = torch.tensor(frame[TARGETS].to_numpy(np.float32),device=device)
    weights = targets.new_tensor(WEIGHTS)
    signature = {'feature_manifest_sha256':file_sha256(Path(args.cache)/'manifest.json'),
        'head_code_sha256':file_sha256(__file__),
        'fold_table_sha256':file_sha256(args.folds_csv),'folds':args.folds,'seed':args.seed,
        'epochs':args.epochs,'swa_last_epochs':args.swa_last_epochs,'width':512,'dropout':.2,
        'lr':args.lr,'weight_decay':.01,'batch_size':args.batch_size,'precision':args.precision,
        'tiny':args.tiny,'objective':'raw_global_weighted_mse_one_training_scale',
        'selection':'fixed_final_swa_endpoint_not_best_validation_epoch'}
    if shuffle_cache:
        signature.update(shuffle_manifest_sha256=file_sha256(Path(shuffle_cache)/'manifest.json'),
            shuffle_grid=grid, shuffle_probability=probability, shuffle_rng_version='private_per_fold_v1')
    output = Path(args.output);output.mkdir(parents=True,exist_ok=True)
    started = time.monotonic(); parts=[]
    for fold in args.folds:
        folder = output/f'fold_{fold}';folder.mkdir(exist_ok=True)
        latest = folder/'latest.pt'
        saved = torch.load(latest, weights_only=True) if latest.exists() else None
        if saved and saved['signature'] != signature:
            raise ValueError('Head resume configuration changed.')
        if (folder/'metrics.json').exists():
            row = json.loads((folder/'metrics.json').read_text())
            if row['signature'] != signature:
                raise ValueError('Completed head configuration changed.')
            parts.append(pd.read_csv(folder/'predictions.csv',dtype={'group_id':str}));continue
        common.seed_all(args.seed+int(fold))
        training = torch.tensor(np.flatnonzero(frame.fold.to_numpy()!=fold),device=device)
        validation = torch.tensor(np.flatnonzero(frame.fold.to_numpy()==fold),device=device)
        if not len(training) or not len(validation):
            raise ValueError('Requested fold lacks train/validation examples.')
        train_targets = targets[training]
        mean = (train_targets*weights).sum()/len(training)
        scale = (((train_targets-mean).square()*weights).sum()/len(training)).sqrt().clamp_min(1.)
        median = train_targets[:,:3].median(0).values
        model = GatedRegressionHead(features.shape[-1],scale,median).to(device)
        initial_hash = model_state_digest(model)
        precision = configure_fp8(model,args.precision,device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
        swa = AveragedModel(model)
        shuffle_rng = torch.Generator(device=device).manual_seed(args.seed+int(fold)+100000)
        epoch=0;cursor=0;order=None;best=None;loss_total=0.;seen=0;optimizer_steps=0
        if saved:
            model.load_state_dict(saved['model']);swa.load_state_dict(saved['swa'])
            optimizer.load_state_dict(saved['optimizer']);common.restore_rng(saved['rng'])
            epoch=saved['epoch'];cursor=saved['cursor'];order=saved['order'];best=saved['best']
            loss_total=saved['loss_total'];seen=saved['seen']
            optimizer_steps=saved.get('optimizer_steps',epoch*math.ceil(len(training)/args.batch_size)
                +math.ceil(cursor/args.batch_size))
            if shuffle_cache:
                shuffle_rng.set_state(saved['shuffle_rng'].cpu())
        runtime={'gpu':torch.cuda.get_device_name() if device.type=='cuda' else 'CPU tiny test',
                 'precision':precision,'encoder_frozen':True,'predictor_frozen':True}
        config={'signature':signature,'feature_origin':manifest['signature'],'runtime':runtime,
            'raw_scale_training_only_g':float(scale),'median_training_only_g':median.tolist(),
            'train_images':len(training),'validation_images':len(validation),
            'trainable_parameters':sum(p.numel() for p in model.parameters()),
            'initial_model_sha256':initial_hash,
            'train_image_ids':frame.iloc[training.cpu().numpy()].image_path.tolist(),
            'validation_image_ids':frame.iloc[validation.cpu().numpy()].image_path.tolist(),
            'validation_cache':'control_orientation_0',
            'train_groups':int(frame.iloc[training.cpu().numpy()].group_id.nunique()),
            'validation_groups':int(frame.iloc[validation.cpu().numpy()].group_id.nunique())}
        common.write_json(folder/'config.json',config)
        def commit():
            common.atomic_save({'signature':signature,'model':model.state_dict(),'swa':swa.state_dict(),
                'optimizer':optimizer.state_dict(),'rng':common.rng_state(),'epoch':epoch,'cursor':cursor,
                'order':order,'best':best,'loss_total':loss_total,'seen':seen,
                'shuffle_rng':shuffle_rng.get_state(),'optimizer_steps':optimizer_steps,
                'initial_model_sha256':initial_hash},latest)
        commit()  # Actual initial state survives a stop before the first completed epoch.
        last_save=time.monotonic()
        bar=tqdm(total=args.epochs,initial=epoch,desc=f'Augmented head fold {fold}',mininterval=2.)
        while epoch < args.epochs:
            if order is None:
                order=training[torch.randperm(len(training),device=device)].cpu().tolist()
                cursor=0;loss_total=0.;seen=0
            model.train()
            optimizer.param_groups[0]['lr']=args.lr*.5*(1+math.cos(math.pi*epoch/args.epochs))
            while cursor < len(order):
                indices=torch.tensor(order[cursor:cursor+args.batch_size],device=device)
                augmentation=torch.randint(4,(len(indices),),device=device)
                optimizer.zero_grad(set_to_none=True)
                with common.amp(device):
                    chosen=select_training_features(features,shuffled,augmentation,indices,probability,shuffle_rng)
                    prediction=model(chosen)
                    loss=((prediction-targets[indices]).square()*weights).sum(1).mean()/scale.square()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite head loss.')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();optimizer_steps+=1
                cursor+=len(indices);seen+=len(indices);loss_total+=float(loss.detach())*len(indices)
                if time.monotonic()-last_save >= 60:
                    commit();last_save=time.monotonic()
                if common.STOP_REQUESTED or time.monotonic()-started>args.max_hours*3600-60:
                    commit();bar.close();return
            if epoch >= args.epochs-args.swa_last_epochs:
                swa.update_parameters(model)
            diagnostic=None
            if (epoch+1)%10==0 or epoch+1==args.epochs:
                model.eval()
                with torch.no_grad(),common.amp(device):
                    raw=model(features[0,validation]).float().cpu().numpy()
                score=scores(targets[validation].cpu().numpy(),raw)['global_weighted_r2_raw']
                diagnostic=score
                if best is None or score>best:
                    best=score;common.atomic_save({'signature':signature,'model':model.state_dict(),
                        'epoch':epoch+1,'score':score,'selection':'diagnostic_only'},folder/'best_diagnostic.pt')
            common.log(folder/'train_log.jsonl',{'epoch':epoch+1,'loss':loss_total/seen,
                'global_weighted_r2_raw_diagnostic':diagnostic,'swa_epochs':int(swa.n_averaged),
                'lr':optimizer.param_groups[0]['lr'],'optimizer_steps':optimizer_steps})
            epoch+=1;cursor=0;order=None;commit();last_save=time.monotonic();bar.update(1)
        evaluator=swa.module if int(swa.n_averaged) else model
        evaluator.eval()
        with torch.no_grad(),common.amp(device):
            prediction=evaluator(features[0,validation]).float().cpu().numpy()
        result=frame.iloc[validation.cpu().numpy()][['image_path','group_id','fold']+TARGETS].copy()
        # CSV round trips retain the exact float32 values used for scoring, rather
        # than pandas' shortened float32 formatting changing the independent score.
        for j,name in enumerate(TARGETS):result[f'pred_{name}']=prediction[:,j].astype(np.float64)
        result.to_csv(folder/'predictions.csv',index=False)
        common.atomic_save({'signature':signature,'model':evaluator.state_dict(),'feature_origin':manifest['signature'],
                            'epochs':epoch,'swa_epochs':int(swa.n_averaged),
                            'optimizer_steps':optimizer_steps,'initial_model_sha256':initial_hash},folder/'selected.pt')
        row=scores(result[TARGETS].to_numpy(),prediction)
        row.update(status='completed_fixed_endpoint',signature=signature,completed_epochs=epoch,
                   best_validation_diagnostic=best,primary_selection=signature['selection'],
                   optimizer_steps=optimizer_steps,initial_model_sha256=initial_hash)
        common.write_json(folder/'metrics.json',row);parts.append(result);bar.close()
    result=pd.concat(parts).sort_values('image_path');result.to_csv(output/'oof_predictions.csv',index=False)
    truth=result[TARGETS].to_numpy();prediction=result[[f'pred_{t}' for t in TARGETS]].to_numpy()
    row=scores(truth,prediction)
    row.update(status='completed_heads',folds=args.folds,predicted_images=len(result),
        expected_images=len(frame),complete_oof=len(result)==len(frame),signature=signature,
        group_bootstrap=group_bootstrap(truth,prediction,result.group_id.to_numpy()),
        limits='Repeated development folds; one seed; fixed endpoint. No leaderboard or acceptance claim.')
    common.write_json(output/'metrics.json',row);print('FROZEN HEAD RESULT:',json.dumps(row),flush=True)


class FrozenHeadInference(nn.Module):
    """Reconstruct from exact frozen source bytes and the actual selected head."""
    def __init__(self, source, checkpoint):
        super().__init__()
        origin=checkpoint['feature_origin']
        if file_sha256(source)!=origin['source_sha256']:
            raise ValueError('Inference source differs from the cached frozen model.')
        self.representation=origin['representation']
        self.frozen, observed=source_model(source,self.representation,origin['tiny'])
        if any(observed[key]!=origin[key] for key in observed):
            raise ValueError('Inference frozen feature provenance changed.')
        state=checkpoint['model']
        self.head=GatedRegressionHead(origin['dim'],state['scale'],torch.ones(3))
        self.head.load_state_dict(state,strict=True)
        self.eval()

    def forward(self, tiles):
        batch,views=tiles.shape[:2]
        if views!=2:
            raise ValueError('Frozen-head inference retains its two complete views.')
        images=tiles.flatten(0,1)
        with torch.no_grad():
            tokens=(self.frozen(images) if self.representation=='encoder'
                    else self.frozen.patch_predictions(images))
            features=tokens.mean(1).reshape(batch,views,-1)
        return self.head(features)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ('cache','fit'):
        q=sub.add_parser(command)
        q.add_argument('--folds-csv',required=True);q.add_argument('--output',required=True)
        q.add_argument('--precision',choices=['fp8','bf16'],default='fp8')
        q.add_argument('--tiny',action='store_true');q.add_argument('--batch-size',type=int,default=32)
        q.add_argument('--max-hours',type=float,default=1.)
        if command=='cache':
            q.add_argument('--init',required=True);q.add_argument('--representation',choices=['encoder','predictor'],required=True)
            q.add_argument('--workers',type=int,default=2)
            q.add_argument('--shuffle-grid',type=int,choices=[0,4,10],default=0)
        else:
            q.add_argument('--cache',required=True);q.add_argument('--folds',type=int,nargs='+',default=[0,1])
            q.add_argument('--epochs',type=int,default=200);q.add_argument('--swa-last-epochs',type=int,default=40)
            q.add_argument('--seed',type=int,default=42);q.add_argument('--lr',type=float,default=3e-4)
            q.add_argument('--shuffle-cache');q.add_argument('--shuffle-prob',type=float,default=.35)
    args=parser.parse_args()
    if args.batch_size<1 or not math.isfinite(args.max_hours) or args.max_hours<=0:
        parser.error('Positive batch/time budgets required.')
    if args.command=='fit' and (not 0<args.swa_last_epochs<=args.epochs
            or not math.isfinite(args.lr) or args.lr<=0):
        parser.error('Positive finite LR and a valid fixed SWA window required.')
    common.STOP_REQUESTED=False
    signal.signal(signal.SIGTERM,common.interrupt_training)
    signal.signal(signal.SIGINT,common.interrupt_training)
    (cache_features if args.command=='cache' else fit_heads)(args)


if __name__=='__main__':main()
