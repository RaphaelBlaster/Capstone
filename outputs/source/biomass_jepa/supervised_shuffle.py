"""Matched online augmentation with supervised final-block I-JEPA adaptation.

Checkpoints retain only the trainable encoder replacement tensors and head. The
unchanged encoder prefix is reconstructed from the exact grass source SHA.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from . import train as common
from .block_shuffle import shuffle_square_tile
from .data import TARGETS, WEIGHTS, file_sha256, normalize_transform
from .metrics import scores
from .models import configure_fp8
from .official_heads import (GatedRegressionHead, source_model, require_runtime,
                             validate_frame, feature_code_revision, model_state_digest)

SOURCE_SHA='6d2b5d9c83423155b4a948822e846caaa9a6bc6be20e3fab13ca06b8e705f43d'
AUGMENTATION='online_stateless_image_epoch_v1'
NATIVE_TARGETS=['dry_grass','dry_clover','dry_weeds','dry_total']
NATIVE_UNITS='native supplied dry-mass units; no CSI target conversion'
NATIVE_AUGMENTATION='full_original_single_view_quarter_turns_no_shuffle_v1'
NATIVE_SCHEDULE='one_full_aux_batch_per_csi_update_cyclic_epoch_order_v1'


def _seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'little')


class OnlineBiomassImages(Dataset):
    """Two complete landscape halves; stateless shuffle and paired quarter turns.

    No global random state is consumed, including in loader workers. Validation
    overrides all augmentation settings to identity. Epoch is part of each seed.
    """
    def __init__(self,frame,size,training=False,grid=0,probability=0.,seed=42,fold=0,epoch=0):
        if grid not in (0,4) or not math.isfinite(probability) or not 0<=probability<=1:
            raise ValueError('Need grid0/4 and finite shuffle probability in[0,1].')
        self.frame=frame.reset_index(drop=True)
        self.transform=normalize_transform(size,train=False)
        self.training=training;self.grid=grid;self.probability=probability
        self.seed=seed;self.fold=fold;self.epoch=epoch

    def __len__(self):return len(self.frame)

    def augmentation_identity(self,index):
        row=self.frame.iloc[index]
        base=(AUGMENTATION,str(row.image_sha256),self.seed,self.fold,self.epoch)
        return {'version':AUGMENTATION,'epoch':self.epoch,'image_sha256':str(row.image_sha256),
            'rotation':random.Random(_seed(*base,'rotation')).randrange(4) if self.training else 0,
            'use_shuffle':bool(self.training and self.grid and random.Random(_seed(*base,'choice')).random()<self.probability),
            'permutation_seeds':[_seed(*base,'permutation',self.grid,view) for view in range(2)]}

    def __getitem__(self,index):
        row=self.frame.iloc[index];identity=self.augmentation_identity(index)
        with Image.open(row.absolute_path) as raw:
            image=ImageOps.exif_transpose(raw).convert('RGB')
            width,height=image.size
            if width!=2*height:
                raise ValueError('Expected complete landscape field with two nonoverlapping square halves.')
            tiles=[]
            for view in range(2):
                tile=image.crop((view*height,0,(view+1)*height,height))
                if identity['use_shuffle']:
                    tile=shuffle_square_tile(tile,self.grid,identity['permutation_seeds'][view])
                tiles.append(self.transform(tile))
        value=torch.rot90(torch.stack(tiles),identity['rotation'],(-2,-1)).contiguous()
        return value,torch.tensor(row[TARGETS].to_numpy(np.float32)),index


class NativeMassImages(Dataset):
    """Full harvested-scene photographs; retain all area, never spatially shuffle."""
    def __init__(self,frame,size,seed=42,fold=0,epoch=0):
        self.frame=frame.reset_index(drop=True);self.transform=normalize_transform(size,train=False)
        self.seed=seed;self.fold=fold;self.epoch=epoch

    def __len__(self):return len(self.frame)

    def augmentation_identity(self,index):
        return {'rotation':random.Random(_seed(NATIVE_AUGMENTATION,self.frame.iloc[index].image_sha256,
            self.seed,self.fold,self.epoch)).randrange(4),'use_shuffle':False,'epoch':self.epoch}

    def __getitem__(self,index):
        row=self.frame.iloc[index]
        with Image.open(row.absolute_path) as raw:
            tile=self.transform(ImageOps.exif_transpose(raw).convert('RGB'))
        tile=torch.rot90(tile,self.augmentation_identity(index)['rotation'],(-2,-1)).contiguous()
        return tile[None],torch.tensor(row[NATIVE_TARGETS].to_numpy(np.float32)),index


class NativeMassHead(GatedRegressionHead):
    """Source-native grass/clover/weeds mass primitives and their total."""
    def forward(self,features):
        primitives=super().forward(features)[:,:3]
        return torch.cat((primitives,primitives.sum(1,keepdim=True)),1)


def validate_auxiliary(path,csi_frame,tiny):
    frame=pd.read_csv(path)
    required=['image_path','absolute_path','image_sha256',*NATIVE_TARGETS]
    if not set(required)<=set(frame) or frame[required].isna().any().any() or not len(frame):
        raise ValueError('Complete native auxiliary labels/identities required.')
    if (frame.image_path.duplicated().any() or frame.image_sha256.duplicated().any()
            or (not tiny and len(frame)!=261)):
        raise ValueError('Exact unique native training photo cohort required.')
    y=frame[NATIVE_TARGETS].to_numpy(float)
    if not np.isfinite(y).all() or (y<0).any() or not np.allclose(y[:,:3].sum(1),y[:,3],rtol=1e-10,atol=1e-9):
        raise ValueError('Native labels must be finite nonnegative and sum to supplied total.')
    if set(frame.image_sha256)&set(csi_frame.image_sha256):
        raise ValueError('Native/CSI byte-identical image overlap is prohibited.')
    for path,digest in zip(frame.absolute_path,frame.image_sha256):
        if file_sha256(path)!=digest:raise ValueError('Native auxiliary image bytes changed.')
    return frame


def auxiliary_order(count,seed,fold,epoch,cycle):
    generator=torch.Generator().manual_seed(_seed(NATIVE_SCHEDULE,seed,fold,epoch,cycle)%(2**63))
    return torch.randperm(count,generator=generator).tolist()


def auxiliary_indices(order,cursor,cycle,count,batch_size,seed,fold,epoch):
    """Fill exactly one native batch, cyclically reshuffling without global RNG."""
    chosen=[]
    while len(chosen)<batch_size:
        if cursor==count:
            cycle+=1;order=auxiliary_order(count,seed,fold,epoch,cycle);cursor=0
        end=min(count,cursor+batch_size-len(chosen));chosen.extend(order[cursor:end]);cursor=end
    return chosen,order,cursor,cycle


class SupervisedEncoderHead(nn.Module):
    def __init__(self,encoder,head,last_blocks,aux_head=None):
        super().__init__();self.encoder=encoder;self.head=head;self.last_blocks=last_blocks;self.aux_head=aux_head

    def train(self,mode=True):
        super().train(mode)
        self.encoder.backbone.patch_embed.eval()
        for block in self.encoder.backbone.blocks[:-self.last_blocks]:block.eval()
        return self

    def forward(self,tiles,auxiliary=False):
        tokens=self.encoder(tiles.flatten(0,1))
        features=tokens.mean(1).reshape(len(tiles),tiles.shape[1],self.encoder.dim)
        if auxiliary:
            if self.aux_head is None:raise ValueError('No native auxiliary head in this checkpoint.')
            return self.aux_head(features)
        return self.head(features)


def _trainable(model):return {name:p for name,p in model.named_parameters() if p.requires_grad}


def _copy_params(model):
    return {name:p.detach().clone() for name,p in _trainable(model).items()}


def _load_params(model,state):
    params=_trainable(model)
    if params.keys()!=state.keys():raise ValueError('Trainable checkpoint scope changed.')
    with torch.no_grad():
        for name,p in params.items():
            if state[name].shape!=p.shape:raise ValueError('Trainable checkpoint tensor shape differs.')
            p.copy_(state[name])


def _delta(model):
    return {name:p.detach() for name,p in model.encoder.named_parameters() if p.requires_grad}


def _digest(params):
    h=hashlib.sha256()
    for name,p in sorted(params.items()):
        h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def reconstruct(source,checkpoint):
    origin=checkpoint['source_origin']
    if file_sha256(source)!=origin['source_sha256']:raise ValueError('Checkpoint frozen source bytes changed.')
    encoder,actual=source_model(source,'encoder',origin['tiny'])
    if actual!=origin:raise ValueError('Checkpoint source metadata changed.')
    blocks=checkpoint['last_blocks'];encoder.configure_trainable(blocks)
    expected={name for name,p in encoder.named_parameters() if p.requires_grad}
    if expected!=set(checkpoint['encoder_delta']):raise ValueError('Checkpoint encoder delta scope differs.')
    encoder.load_state_dict(checkpoint['encoder_delta'],strict=False)
    head_state=checkpoint['head']
    head=GatedRegressionHead(origin['dim'],head_state['scale'],torch.ones(3))
    head.load_state_dict(head_state,strict=True)
    aux_state=checkpoint.get('aux_head');aux_head=None
    if aux_state is not None:
        aux_head=NativeMassHead(origin['dim'],aux_state['scale'],torch.ones(3))
        aux_head.load_state_dict(aux_state,strict=True)
    return SupervisedEncoderHead(encoder,head,blocks,aux_head)


def _predict(model,dataset,device,batch_size,workers):
    model.eval();values=[]
    generator=torch.Generator().manual_seed(0)
    loader=DataLoader(dataset,batch_size=batch_size,num_workers=workers,shuffle=False,generator=generator)
    with torch.no_grad(),common.amp(device):
        for tiles,_,_ in loader:values.append(model(tiles.to(device)).float().cpu().numpy().astype(np.float64))
    return np.concatenate(values)


def _aggregate(output,frame,args,signature):
    parts=[];completed=[]
    for fold in args.folds:
        folder=output/f'fold_{fold}';path=folder/'metrics.json'
        if not path.exists():continue
        metric=json.loads(path.read_text())
        if metric['signature']!=signature:raise ValueError('Completed fold configuration differs.')
        if metric['status']=='completed_fixed_endpoint':
            parts.append(pd.read_csv(folder/'predictions.csv',dtype={'group_id':str}));completed.append(fold)
    if parts:
        prediction=pd.concat(parts).sort_values('image_path')
        prediction.to_csv(output/'oof_predictions.csv',index=False)
        result=scores(prediction[TARGETS].to_numpy(),prediction[['pred_'+t for t in TARGETS]].to_numpy())
    else:prediction=pd.DataFrame();result={}
    result.update(status='completed_heads' if completed==args.folds else 'incomplete_heads',
        signature=signature,folds=args.folds,completed_folds=completed,predicted_images=len(prediction),
        expected_images=len(frame),complete_oof=False,
        limits='Reused partial development folds; one seed; no competition or protected-confirmation claim.')
    common.write_json(output/'metrics.json',result)
    return result


def fit(args):
    aux_csv=getattr(args,'aux_folds_csv',None);aux_weight=getattr(args,'aux_weight',.25)
    if args.folds not in ([0,1],[0,1,2]):raise ValueError('Exactly two or three original folds required.')
    if args.run_fold is not None and args.run_fold not in args.folds:raise ValueError('run-fold must be authorized.')
    if (args.last_blocks!=2 or args.epochs<1 or not 1<=args.swa_last_epochs<=args.epochs
            or args.batch_size<1 or args.workers<0 or args.encoder_lr<=0 or args.head_lr<=0):
        raise ValueError('Invalid matched final-two-block training recipe.')
    if args.shuffle_grid not in (0,4) or not math.isfinite(args.shuffle_probability) or not 0<=args.shuffle_probability<=1:
        raise ValueError('Invalid shuffle grid/probability.')
    if not args.tiny and (args.epochs!=200 or args.swa_last_epochs!=40 or args.batch_size!=16
            or args.encoder_lr!=5e-6 or args.head_lr!=1e-4
            or (args.shuffle_grid,args.shuffle_probability) not in ((0,0.),(4,.2))
            or not aux_csv or aux_weight!=.25):
        raise ValueError('Real experiment requires the exact authorized matched recipe.')
    device=require_runtime(args);frame=validate_frame(args.folds_csv)
    if not math.isfinite(aux_weight) or aux_weight<=0:raise ValueError('Positive finite auxiliary weight required.')
    aux_frame=validate_auxiliary(aux_csv,frame,args.tiny) if aux_csv else None
    source_sha=file_sha256(args.init)
    if not args.tiny and source_sha!=SOURCE_SHA:raise ValueError('Exact saved grass student source required.')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    signature={'source_sha256':source_sha,'fold_table_sha256':file_sha256(args.folds_csv),
        'code_revision':feature_code_revision(),'folds':args.folds,'seed':args.seed,'epochs':args.epochs,
        'swa_last_epochs':args.swa_last_epochs,'last_blocks':args.last_blocks,'width':512,'dropout':.2,
        'encoder_lr':args.encoder_lr,'head_lr':args.head_lr,'weight_decay':.01,'batch_size':args.batch_size,
        'precision':args.precision,'tiny':args.tiny,'shuffle_grid':args.shuffle_grid,
        'shuffle_probability':args.shuffle_probability,'augmentation':AUGMENTATION,
        'objective':'raw_global_weighted_mse_one_training_scale',
        'selection':'fixed_final_swa_endpoint_not_best_validation_epoch'}
    if aux_frame is not None:
        signature.update(aux_csv_sha256=file_sha256(aux_csv),aux_weight=aux_weight,
            aux_targets=NATIVE_TARGETS,aux_augmentation=NATIVE_AUGMENTATION,aux_schedule=NATIVE_SCHEDULE)
    started=time.monotonic()
    active=[args.run_fold] if args.run_fold is not None else args.folds
    for fold in active:
        folder=output/f'fold_{fold}';folder.mkdir(exist_ok=True)
        if (folder/'metrics.json').exists():
            metric=json.loads((folder/'metrics.json').read_text())
            if metric['signature']!=signature:raise ValueError('Completed fold configuration differs.')
            continue
        common.seed_all(args.seed+fold)
        encoder,origin=source_model(args.init,'encoder',args.tiny)
        if origin['grass_steps']!=2400:raise ValueError('Completed grass adaptation required; no new SSL.')
        encoder.configure_trainable(args.last_blocks)
        training=np.flatnonzero(frame.fold.to_numpy()!=fold);validation=np.flatnonzero(frame.fold.to_numpy()==fold)
        if not len(training) or not len(validation):raise ValueError('Empty fold partition.')
        y=torch.tensor(frame.iloc[training][TARGETS].to_numpy(np.float32))
        weights=torch.tensor(WEIGHTS,dtype=torch.float32)
        mean=(y*weights).sum()/len(y)
        scale=(((y-mean).square()*weights).sum()/len(y)).sqrt().clamp_min(1.)
        median=y[:,:3].median(0).values
        head=GatedRegressionHead(origin['dim'],scale,median)
        aux_head=None;aux_initial=None;aux_scale=None;aux_median=None;aux_data=None;aux_source=None
        if aux_frame is not None:
            aux_y=torch.tensor(aux_frame[NATIVE_TARGETS].to_numpy(np.float32))
            aux_mean=aux_y.mean()
            aux_scale=(aux_y-aux_mean).square().mean().sqrt().clamp_min(1.)
            aux_median=aux_y[:,:3].median(0).values
            aux_head=NativeMassHead(origin['dim'],aux_scale,aux_median)
            aux_initial=model_state_digest(aux_head)
            aux_data=NativeMassImages(aux_frame,origin['size'],seed=args.seed,fold=fold)
            aux_source={'csv_sha256':file_sha256(aux_csv),'rows':len(aux_frame),'weight':aux_weight,
                'images':aux_frame[['image_path','image_sha256']].to_dict('records'),
                'targets':NATIVE_TARGETS,'units':NATIVE_UNITS,'augmentation':NATIVE_AUGMENTATION,
                'schedule':NATIVE_SCHEDULE,'batch_size':args.batch_size,
                'loss':'uniform_four_native_targets_mse_over_own_training_scale_squared'}
        encoder_initial=_digest({n:p for n,p in encoder.named_parameters() if p.requires_grad})
        head_initial=model_state_digest(head)
        model=SupervisedEncoderHead(encoder,head,args.last_blocks,aux_head).to(device)
        precision=configure_fp8(model,args.precision,device)
        encoder_params=[p for p in model.encoder.parameters() if p.requires_grad]
        head_params=list(model.head.parameters())+(list(model.aux_head.parameters()) if model.aux_head is not None else [])
        optimizer=torch.optim.AdamW([{'params':encoder_params,'lr':args.encoder_lr},
            {'params':head_params,'lr':args.head_lr}],weight_decay=.01)
        frozen={n:p for n,p in model.encoder.named_parameters() if not p.requires_grad}
        if not frozen or any(p.requires_grad for p in frozen.values()):raise ValueError('Encoder prefix is not frozen.')
        probe_name,probe=next(iter(frozen.items()));probe_initial=probe.detach().clone()
        dataset=OnlineBiomassImages(frame,origin['size'],training=True,grid=args.shuffle_grid,
            probability=args.shuffle_probability,seed=args.seed,fold=fold)
        validation_data=OnlineBiomassImages(frame.iloc[validation],origin['size'],training=False)
        epoch=0;cursor=0;order=None;loss_total=0.;seen=0;steps=0;best=None;swa={};n_swa=0;proof=None
        aux_cursor=0;aux_cycle=0;aux_order=None;aux_exposures=0;aux_epoch_exposures=0;aux_loss_total=0.
        latest=folder/'latest.pt'
        if latest.exists():
            saved=torch.load(latest,weights_only=True,map_location='cpu')
            if saved['signature']!=signature:raise ValueError('Resume configuration differs.')
            model.encoder.load_state_dict(saved['encoder_delta'],strict=False)
            model.head.load_state_dict(saved['head']);optimizer.load_state_dict(saved['optimizer'])
            if aux_head is not None:
                model.aux_head.load_state_dict(saved['aux_head'])
                aux_cursor=saved['aux_cursor'];aux_cycle=saved['aux_cycle'];aux_order=saved['aux_order']
                aux_exposures=saved['aux_exposures'];aux_epoch_exposures=saved['aux_epoch_exposures']
                aux_loss_total=saved['aux_loss_total']
            epoch=saved['epoch'];cursor=saved['cursor'];order=saved['order'];steps=saved['optimizer_steps']
            loss_total=saved['loss_total'];seen=saved['seen'];best=saved['best'];n_swa=saved['swa_epochs']
            swa={k:v.to(device) for k,v in saved['swa'].items()};proof=saved['first_update_proof']
            common.restore_rng(saved['rng'])
        config={'signature':signature,'source_origin':origin,'runtime':{'precision':precision,
            'gpu':torch.cuda.get_device_name() if device.type=='cuda' else 'CPU tiny test',
            'encoder_learning':True,'predictor_used':False,'ssl_updates':False},
            'encoder_initial_sha256':encoder_initial,'head_initial_sha256':head_initial,
            'raw_scale_training_only_g':float(scale),'median_training_only_g':median.tolist(),
            'train_image_ids':frame.iloc[training].image_path.tolist(),
            'validation_image_ids':frame.iloc[validation].image_path.tolist(),
            'train_images':len(training),'validation_images':len(validation),
            'train_groups':int(frame.iloc[training].group_id.nunique()),
            'validation_groups':int(frame.iloc[validation].group_id.nunique()),
            'validation_augmentation':'identity_only','trainable_encoder_parameters':sum(p.numel() for p in encoder_params),
            'trainable_head_parameters':sum(p.numel() for p in model.head.parameters()),'first_update_proof':proof,
            'aux_source':aux_source,'aux_initial_sha256':aux_initial,
            'aux_scale_training_only_native':float(aux_scale) if aux_scale is not None else None,
            'aux_median_training_only_native':aux_median.tolist() if aux_median is not None else None,
            'trainable_aux_head_parameters':sum(p.numel() for p in model.aux_head.parameters()) if model.aux_head is not None else 0}
        common.write_json(folder/'config.json',config)

        def checkpoint():
            return {'signature':signature,'source_origin':origin,'last_blocks':args.last_blocks,
                'encoder_delta':_delta(model),'head':model.head.state_dict(),
                'encoder_initial_sha256':encoder_initial,'head_initial_sha256':head_initial,
                'first_update_proof':proof,'optimizer_steps':steps,
                'aux_head':model.aux_head.state_dict() if model.aux_head is not None else None,
                'aux_initial_sha256':aux_initial,'aux_exposures':aux_exposures}

        def commit():
            common.atomic_save({**checkpoint(),'optimizer':optimizer.state_dict(),'rng':common.rng_state(),
                'epoch':epoch,'cursor':cursor,'order':order,'loss_total':loss_total,'seen':seen,
                'best':best,'swa':swa,'swa_epochs':n_swa,
                'aux_order':aux_order,'aux_cursor':aux_cursor,'aux_cycle':aux_cycle,
                'aux_epoch_exposures':aux_epoch_exposures,'aux_loss_total':aux_loss_total,
                'augmentation_identity':{'version':AUGMENTATION,'seed':args.seed,'fold':fold,'epoch':epoch}},latest)

        commit();last_save=time.monotonic()
        while epoch<args.epochs:
            dataset.epoch=epoch
            if order is None:
                order=torch.tensor(training)[torch.randperm(len(training))].tolist();cursor=0;loss_total=0.;seen=0
                aux_epoch_exposures=0;aux_loss_total=0.
                if aux_frame is not None:
                    aux_order=auxiliary_order(len(aux_frame),args.seed,fold,epoch,0);aux_cursor=0;aux_cycle=0
            if aux_data is not None:aux_data.epoch=epoch
            model.train()
            factor=.5*(1+math.cos(math.pi*epoch/args.epochs))
            optimizer.param_groups[0]['lr']=args.encoder_lr*factor;optimizer.param_groups[1]['lr']=args.head_lr*factor
            # Local loader RNG prevents worker startup from perturbing dropout/order.
            loader=DataLoader(dataset,batch_size=args.batch_size,sampler=order[cursor:],num_workers=args.workers,
                generator=torch.Generator().manual_seed(_seed(args.seed,fold,epoch,'loader')%(2**63)))
            for tiles,targets,indices in loader:
                optimizer.zero_grad(set_to_none=True)
                with common.amp(device):
                    prediction=model(tiles.to(device))
                    loss=((prediction-targets.to(device)).square()*weights.to(device)).sum(1).mean()/scale.to(device).square()
                    aux_loss=None
                    if aux_data is not None:
                        aux_idx,aux_order,aux_cursor,aux_cycle=auxiliary_indices(aux_order,aux_cursor,aux_cycle,
                            len(aux_frame),args.batch_size,args.seed,fold,epoch)
                        samples=[aux_data[i] for i in aux_idx]
                        aux_tiles=torch.stack([sample[0] for sample in samples]).to(device)
                        aux_targets=torch.stack([sample[1] for sample in samples]).to(device)
                        aux_prediction=model(aux_tiles,auxiliary=True)
                        aux_loss=(aux_prediction-aux_targets).square().mean()/aux_scale.to(device).square()
                        loss=loss+aux_weight*aux_loss
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite supervised loss.')
                loss.backward();torch.nn.utils.clip_grad_norm_([*encoder_params,*head_params],1.,error_if_nonfinite=True)
                optimizer.step();steps+=1;cursor+=len(indices);seen+=len(indices);loss_total+=float(loss.detach())*len(indices)
                if aux_data is not None:
                    aux_exposures+=len(aux_idx);aux_epoch_exposures+=len(aux_idx)
                    aux_loss_total+=float(aux_loss.detach())*len(aux_idx)
                if proof is None:
                    encoder_after=_digest({n:p for n,p in model.encoder.named_parameters() if p.requires_grad})
                    head_after=model_state_digest(model.head)
                    proof={'update':1,'encoder_changed':encoder_after!=encoder_initial,'head_changed':head_after!=head_initial,
                        'encoder_before_sha256':encoder_initial,'encoder_after_sha256':encoder_after,
                        'head_before_sha256':head_initial,'head_after_sha256':head_after,
                        'frozen_prefix_requires_grad_false':all(not p.requires_grad for p in frozen.values()),
                        'frozen_prefix_probe_name':probe_name,'frozen_prefix_probe_unchanged':torch.equal(probe,probe_initial)}
                    if model.aux_head is not None:
                        aux_after=model_state_digest(model.aux_head)
                        proof.update(aux_head_changed=aux_after!=aux_initial,
                            aux_head_before_sha256=aux_initial,aux_head_after_sha256=aux_after)
                    if not all(proof[k] for k in ('encoder_changed','head_changed','frozen_prefix_requires_grad_false','frozen_prefix_probe_unchanged')):
                        raise RuntimeError('First supervised optimizer update failed encoder/head/frozen-prefix proof.')
                    if model.aux_head is not None and not proof['aux_head_changed']:
                        raise RuntimeError('First supervised optimizer update failed native auxiliary head proof.')
                    config['first_update_proof']=proof;common.write_json(folder/'config.json',config)
                    commit();last_save=time.monotonic()
                if time.monotonic()-last_save>=60:commit();last_save=time.monotonic()
                if common.STOP_REQUESTED or time.monotonic()-started>args.max_hours*3600-60:
                    commit();return _aggregate(output,frame,args,signature)
            if epoch>=args.epochs-args.swa_last_epochs:
                if not swa:swa=_copy_params(model)
                else:
                    with torch.no_grad():
                        for name,p in _trainable(model).items():swa[name].add_((p.detach()-swa[name])/(n_swa+1))
                n_swa+=1
            diagnostic=None
            if (epoch+1)%10==0 or epoch+1==args.epochs:
                prediction=_predict(model,validation_data,device,args.batch_size,args.workers)
                diagnostic=scores(frame.iloc[validation][TARGETS].to_numpy(),prediction)['global_weighted_r2_raw']
                if best is None or diagnostic>best:
                    best=diagnostic
                    common.atomic_save({**checkpoint(),'epochs':epoch+1,'selection':'diagnostic_only'},folder/'best_diagnostic.pt')
            common.log(folder/'train_log.jsonl',{'epoch':epoch+1,'loss':loss_total/seen,
                'global_weighted_r2_raw_diagnostic':diagnostic,'encoder_lr':optimizer.param_groups[0]['lr'],
                'head_lr':optimizer.param_groups[1]['lr'],'optimizer_steps':steps,'swa_epochs':n_swa,
                'aux_exposures':aux_exposures,'aux_epoch_exposures':aux_epoch_exposures,
                'aux_loss':aux_loss_total/aux_epoch_exposures if aux_epoch_exposures else None})
            epoch+=1;cursor=0;order=None;commit();last_save=time.monotonic()
        if n_swa!=args.swa_last_epochs:raise RuntimeError('Fixed SWA endpoint incomplete.')
        _load_params(model,swa)
        prediction=_predict(model,validation_data,device,args.batch_size,args.workers)
        result=frame.iloc[validation][['image_path','group_id','fold']+TARGETS].copy()
        for j,name in enumerate(TARGETS):result['pred_'+name]=prediction[:,j]
        result.to_csv(folder/'predictions.csv',index=False)
        common.atomic_save({**checkpoint(),'epochs':epoch,'swa_epochs':n_swa},folder/'selected.pt')
        metric=scores(result[TARGETS].to_numpy(),prediction)
        metric.update(status='completed_fixed_endpoint',signature=signature,completed_epochs=epoch,swa_epochs=n_swa,
            optimizer_steps=steps,encoder_initial_sha256=encoder_initial,head_initial_sha256=head_initial,
            first_update_proof=proof,primary_selection=signature['selection'],
            aux_initial_sha256=aux_initial,aux_exposures=aux_exposures)
        common.write_json(folder/'metrics.json',metric)
        del model,optimizer,encoder,head,swa
    return _aggregate(output,frame,args,signature)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--init',required=True);p.add_argument('--folds-csv',required=True);p.add_argument('--output',required=True)
    p.add_argument('--folds',nargs='+',type=int,default=[0,1,2]);p.add_argument('--run-fold',type=int)
    p.add_argument('--shuffle-grid',type=int,choices=[0,4],default=0)
    p.add_argument('--shuffle-probability','--shuffle-prob',type=float,default=0.)
    p.add_argument('--last-blocks','--unfreeze-last',type=int,default=2)
    p.add_argument('--encoder-lr',type=float,default=5e-6);p.add_argument('--head-lr',type=float,default=1e-4)
    p.add_argument('--epochs',type=int,default=200);p.add_argument('--swa-last-epochs',type=int,default=40)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--max-hours',type=float,default=1.);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--aux-folds-csv');p.add_argument('--aux-weight',type=float,default=.25)
    p.add_argument('--tiny',action='store_true');p.add_argument('--precision',choices=['fp8','bf16'],default='fp8')
    args=p.parse_args()
    signal.signal(signal.SIGTERM,common.interrupt_training);signal.signal(signal.SIGINT,common.interrupt_training)
    print(json.dumps(fit(args)),flush=True)


if __name__=='__main__':main()
