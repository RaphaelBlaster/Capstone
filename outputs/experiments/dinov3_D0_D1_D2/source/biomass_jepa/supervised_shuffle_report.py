"""Independent paired analysis for supervised encoder-tail online shuffle training.

This does not call the frozen-head protocol audit. Each reported fold must prove
encoder/head updates, an unchanged frozen prefix, and the same fixed SWA endpoint.
"""
from __future__ import annotations
import argparse
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from . import block_shuffle_report as shared

AUX_TARGETS=['dry_grass','dry_clover','dry_weeds','dry_total']
ARMS=('control','shuffle_4')
FOLDS=(0,1,2)
TARGETS=shared.TARGETS
PREDICTED=shared.PREDICTED
LIMITS=('Supervised learning in the final two encoder blocks and final norm plus the gated head. '
        'Reused partial development cohort; one seed; dates are proxies, not verified physical sites. '
        'No protected confirmation, full OOF, leaderboard result, new SSL or predictor. '
        'Bootstrap conditions on fixed trained models and excludes training-seed uncertainty and prior model-selection bias. '
        'Online synthetic permutations add no independent biological measurements. Benefit and journal acceptance are not guaranteed.')


def prepare_native_auxiliary(raw_csv,image_root,output,excluded_sha256,expected_rows=261):
    """Prepare original full-scene native labels; never manufacture CSIRO targets."""
    from PIL import Image,ImageOps
    raw_csv,image_root,output=Path(raw_csv),Path(image_root).resolve(),Path(output)
    native=pd.read_csv(raw_csv,sep=';',encoding='utf-8-sig')
    if len(native)!=expected_rows or not {'image_file_name',*AUX_TARGETS}.issubset(native):
        raise ValueError('Native auxiliary count/required dry-mass columns changed.')
    if native.image_file_name.isna().any() or native.image_file_name.duplicated().any():
        raise ValueError('Native auxiliary image names are missing/duplicated.')
    values=native[AUX_TARGETS].to_numpy(float)
    if not np.isfinite(values).all() or (values<0).any() or (values[:,3]<=0).any():
        raise ValueError('Native component labels must be nonnegative and totals positive; retain measured zeros.')
    residual=values[:,3]-values[:,:3].sum(1)
    if not np.allclose(residual,0,rtol=0,atol=1e-8):raise ValueError('Native grass/clover/weeds sum differs from unchanged dry total.')
    prepared=native[['image_file_name']+AUX_TARGETS].rename(columns={'image_file_name':'image_path'}).copy()
    hashes=[];paths=[];geometry=None
    for name in prepared.image_path:
        path=(image_root/str(name)).resolve()
        if not path.is_relative_to(image_root) or not path.is_file():raise ValueError('Missing or escaping native original photograph.')
        digest=shared._sha(path)
        if digest in excluded_sha256:raise ValueError('Native supervised image has exact overlap with CSIRO or historical SSL corpus.')
        with Image.open(path) as raw:
            scene=ImageOps.exif_transpose(raw)
            if geometry is None:geometry=list(scene.size)
            if min(scene.size)<1:raise ValueError('Invalid native full-scene geometry.')
        hashes.append(digest);paths.append(str(path))
    if len(hashes)!=len(set(hashes)):raise ValueError('Duplicate byte-identical native supervised photographs.')
    prepared['image_sha256']=hashes;prepared['absolute_path']=paths
    prepared['units']='native dry-mass units as provided; no conversion asserted'
    output.parent.mkdir(parents=True,exist_ok=True);prepared.to_csv(output,index=False)
    return dict(rows=len(prepared),targets=AUX_TARGETS,raw_csv_sha256=shared._sha(raw_csv),
        prepared_csv_sha256=shared._sha(output),units=prepared.units.iloc[0],scene_geometry_first=geometry,
        max_abs_label_sum_residual=float(np.abs(residual).max()),exact_overlap=0,
        scene_transform='One full original scene resized to224; no crop, no shuffle; paired quarter-turns',
        images=prepared[['image_path','image_sha256']].to_dict('records'),
        source_dataset='pushpakbhoge/grass-clover-datasets/1',evaluation='CSIRO only; no auxiliary holdout score claimed')


def _load_folds(root,frame):
    loaded={arm:{} for arm in ARMS}
    information={}
    for arm in ARMS:
        pending=[]
        for fold in FOLDS:
            directory=root/'arms'/arm/f'fold_{fold}'
            metrics=shared._read_json(directory/'metrics.json')
            predictions=directory/'predictions.csv'
            if predictions.is_file():
                pred=shared._validate_predictions(pd.read_csv(predictions,dtype={'group_id':str}),frame)
                if not np.array_equal(pred.image_path.to_numpy(),frame[frame.fold==fold].image_path.to_numpy()):
                    raise ValueError('Saved fold predictions do not cover the exact original validation cohort once.')
                shared._check_saved_score(metrics,pred)
            else:
                pred=None
            complete=(metrics.get('status')=='completed_fixed_endpoint' and metrics.get('completed_epochs')==200
                      and metrics.get('swa_epochs')==40 and (directory/'selected.pt').is_file() and pred is not None)
            if complete:
                config=shared._read_json(directory/'config.json')
                if not config:raise ValueError('Completed supervised fold lacks config evidence.')
                loaded[arm][fold]=(pred,config,metrics)
            elif directory.is_dir():
                pending.append(fold)
        completed=sorted(loaded[arm])
        info=dict(status='completed_authorized_folds' if completed==list(FOLDS) else 'partial_fixed_endpoint' if completed else 'incomplete',
                  completed_folds=completed,incomplete_or_unverified_folds=pending)
        if completed:
            pred=pd.concat([loaded[arm][fold][0] for fold in completed]).sort_values('image_path').reset_index(drop=True)
            info.update(completed_saved_images=len(pred),completed_saved_global_weighted_r2_raw=shared.raw_weighted_r2(pred[TARGETS],pred[PREDICTED]))
            pooled_path=root/'arms'/arm/'oof_predictions.csv'
            if pooled_path.is_file():
                pooled=shared._validate_predictions(pd.read_csv(pooled_path,dtype={'group_id':str}),frame)
                columns=['image_path','group_id','fold']+TARGETS+PREDICTED
                if not pooled[columns].equals(pred[columns]):raise ValueError('Saved pooled rows differ from completed supervised fold outputs.')
                shared._check_saved_score(shared._read_json(root/'arms'/arm/'metrics.json'),pooled)
        information[arm]=info
    return loaded,information


def _audit(root,frame,loaded,paired_folds):
    folds_audit={};source_hashes=set();tiny_flags=[]
    common_fields=['encoder_initial_sha256','head_initial_sha256','raw_scale_training_only_g','median_training_only_g',
                   'train_image_ids','validation_image_ids','validation_augmentation']
    expected=dict(folds=list(FOLDS),seed=42,epochs=200,swa_last_epochs=40,width=512,dropout=.2,
        batch_size=16,encoder_lr=5e-6,head_lr=1e-4,last_blocks=2,
        objective='raw_global_weighted_mse_one_training_scale',selection='fixed_final_swa_endpoint_not_best_validation_epoch',
        augmentation='online_stateless_image_epoch_v1')
    complete_logs=True
    for fold in sorted(set().union(*(set(parts) for parts in loaded.values()))):
        configs={arm:loaded[arm][fold][1] for arm in ARMS if fold in loaded[arm]}
        baseline=next(iter(configs.values()));curves={}
        for arm,config in configs.items():
            metrics=loaded[arm][fold][2]
            if any(field not in config for field in common_fields):raise ValueError('Missing supervised initialization/split/scale evidence.')
            if any(config[field]!=baseline[field] for field in common_fields):raise ValueError('Paired supervised initialization, scales or splits differ.')
            if (set(config['train_image_ids'])!=set(frame[frame.fold!=fold].image_path)
                    or set(config['validation_image_ids'])!=set(frame[frame.fold==fold].image_path)
                    or len(config['train_image_ids'])!=len(set(config['train_image_ids']))):
                raise ValueError('Supervised training/validation IDs differ from unchanged biological fold splits.')
            if config['validation_augmentation']!='identity_only':raise ValueError('Supervised validation must be unshuffled identity input.')
            signature=config.get('signature',{})
            if metrics.get('signature')!=signature:raise ValueError('Saved supervised signature differs from config.')
            if any(signature.get(k)!=v for k,v in expected.items()):raise ValueError('Supervised fixed recipe differs.')
            grid,probability=(0,0.) if arm=='control' else (4,.2)
            if signature.get('shuffle_grid')!=grid or signature.get('shuffle_probability')!=probability:
                raise ValueError('Supervised arm differs from control/4×4p0.2 specification.')
            if {k:v for k,v in signature.items() if not k.startswith('shuffle_')}!={k:v for k,v in baseline['signature'].items() if not k.startswith('shuffle_')}:
                raise ValueError('Paired supervised base signatures differ.')
            origin=config.get('source_origin',{})
            if origin.get('representation')!='encoder' or origin.get('grass_steps')!=2400 or not origin.get('source_sha256'):
                raise ValueError('Need verified completed grass-adapted encoder source identity.')
            if origin!=baseline.get('source_origin'):raise ValueError('Supervised paired source origins differ.')
            source_hashes.add(origin['source_sha256'])
            proof=config.get('first_update_proof',{})
            if proof.get('update')!=1 or any(proof.get(flag) is not True for flag in ['encoder_changed','head_changed','frozen_prefix_requires_grad_false','frozen_prefix_probe_unchanged']):
                raise ValueError('Missing encoder/head update or frozen-prefix proof.')
            if metrics.get('first_update_proof')!=proof:raise ValueError('Supervised update proof differs between config and metrics.')
            for part in ['encoder','head']:
                if part+'_before_sha256' in proof and proof[part+'_before_sha256']!=config[part+'_initial_sha256']:
                    raise ValueError('First update before digest differs from paired initialization.')
                if part+'_after_sha256' in proof and proof.get(part+'_after_sha256')==proof.get(part+'_before_sha256'):
                    raise ValueError('First supervised update did not change trainable parameters.')
            batch=signature['batch_size'];per_epoch=(len(config['train_image_ids'])+batch-1)//batch
            if metrics.get('optimizer_steps')!=200*per_epoch:raise ValueError('Supervised optimizer counts differ from fixed matched exposure.')
            runtime=config.get('runtime',{})
            if runtime.get('encoder_learning') is not True or runtime.get('predictor_used') is not False or runtime.get('ssl_updates',False) is not False:
                raise ValueError('Supervised runtime must update encoder without predictor/new SSL.')
            tiny=signature.get('tiny') is True and origin.get('tiny') is True;tiny_flags.append(tiny)
            if not tiny:
                precision=runtime.get('precision',{})
                gpu=''.join(c for c in str(runtime.get('gpu','')).lower() if c.isalnum())
                if (not isinstance(precision,dict) or signature.get('precision')!='fp8' or precision.get('requested')!='fp8'
                        or precision.get('active')!='fp8_linear_bf16_other' or not isinstance(precision.get('float8_layers'),int)
                        or precision['float8_layers']<=0 or 'rtxpro6000' not in gpu or 'blackwell' not in gpu):
                    raise ValueError('Real supervised learning requires actual RTX PRO6000 Blackwell and active FP8 linears.')
            path=root/'arms'/arm/f'fold_{fold}'/'train_log.jsonl'
            rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []
            complete=(len(rows)==200 and [r.get('epoch') for r in rows]==list(range(1,201))
                      and all(all(key in r for key in ['encoder_lr','head_lr','optimizer_steps']) for r in rows))
            if complete:
                if any(r['optimizer_steps']!=r['epoch']*per_epoch or not np.isfinite(r['encoder_lr']) or not np.isfinite(r['head_lr']) for r in rows):
                    raise ValueError('Invalid supervised per-epoch optimizer/LR evidence.')
                curves[arm]=[(r['epoch'],r['encoder_lr'],r['head_lr'],r['optimizer_steps']) for r in rows]
        if len(curves)!=len(configs):complete_logs=False
        if len(curves)==len(configs) and any(curve!=next(iter(curves.values())) for curve in curves.values()):
            raise ValueError('Paired supervised learning-rate/update curves differ.')
        folds_audit[str(fold)]=dict(paired=fold in paired_folds,matched_complete_lr_logs=len(curves)==len(configs),
                                   encoder_and_head_update_proof=True,frozen_prefix_proof=True)
    if len(source_hashes)>1:raise ValueError('Different supervised source checkpoints across folds.')
    return dict(status='verified_reported_protocol' if complete_logs and folds_audit else 'incomplete_provenance',
        folds=folds_audit,source_sha256=sorted(source_hashes),tiny_software_smoke=bool(tiny_flags) and all(tiny_flags),
        scope='Verifies saved configs/logs and prediction identities; checkpoint tensors and remote hardware are not reexecuted by analysis.')


def _architecture(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,3));ax.axis('off')
    nodes=['Original field\n2000×1000','Two original1000²halves\nTraining:4×4p0.2online\nthen224resize/quarter-turn',
           'Grass I-JEPA student\nLast2blocks+finalnorm learn\nPrefix remains frozen','Same gated head learns\nThree positive primitives\nFive biomass outputs']
    for i,label in enumerate(nodes):
        x=.12+i*.25
        ax.text(x,.65,label,ha='center',va='center',transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=.6',facecolor='#edf3f6',edgecolor='#526976'))
        if i<3:ax.annotate('',xy=(x+.17,.65),xytext=(x+.08,.65),xycoords='axes fraction',arrowprops=dict(arrowstyle='->'))
    ax.text(.5,.12,'Validation: original unshuffled identity input; same learned encoder/head. No predictor or new SSL.',ha='center',transform=ax.transAxes)
    fig.suptitle('Supervised online permutation: encoder tail and head learn jointly')
    path=root/'figures/figure_01_architecture'
    fig.savefig(path.with_suffix('.svg'),bbox_inches='tight');fig.savefig(path.with_suffix('.png'),dpi=300,bbox_inches='tight');plt.close(fig)
    shared._json(root/'figures/figure_01_architecture_source.json',dict(nodes=nodes,encoder_learning=True,encoder_last_blocks=2,
        final_norm_trainable=True,shuffle_probability=.2,predictor_used=False,new_ssl=False,validation_shuffle=False))


def _examples(root,frame):
    """Only two arms; disclose synthetic4×4rearrangements, never a fine-shuffle arm."""
    import matplotlib.pyplot as plt
    from PIL import Image,ImageOps
    from .block_shuffle import permutation_seed,shuffle_square_tile
    status=dict(status='missing',reason='Fewer than two verified original photos available.')
    sources=[]
    if 'absolute_path' in frame:
        candidates=frame[frame.absolute_path.map(lambda p:isinstance(p,str) and Path(p).is_file())].iloc[:3]
        if len(candidates)>=2:
            fig,axes=plt.subplots(len(candidates),2,figsize=(10,3*len(candidates)),squeeze=False)
            for i,(_,row) in enumerate(candidates.iterrows()):
                if shared._sha(row.absolute_path)!=row.image_sha256:raise ValueError('Supervised example photo bytes changed.')
                with Image.open(row.absolute_path) as raw:image=ImageOps.exif_transpose(raw).convert('RGB')
                if image.size!=(2000,1000):raise ValueError('Need original full field photos for examples.')
                transformed=image.copy();seeds=[]
                for view in range(2):
                    seed=permutation_seed(row.image_sha256,4,0,view);seeds.append(seed)
                    transformed.paste(shuffle_square_tile(image.crop((view*1000,0,(view+1)*1000,1000)),4,seed),(view*1000,0))
                for col,photo in enumerate([image,transformed]):
                    axes[i,col].imshow(photo);axes[i,col].axis('off')
                    axes[i,col].set_title(f'Original sample index{row.name}: '+('original' if col==0 else 'synthetic4×4rearrangement'))
                sources.append(dict(sample_index=int(row.name),image_sha256=row.image_sha256,seeds=seeds,display_variant=0))
            fig.suptitle('Illustrative synthetic rearrangements; online training uses epoch-varying seeds')
            for ext in ['svg','png']:fig.savefig(root/f'figures/figure_02_examples.{ext}',dpi=300,bbox_inches='tight')
            plt.close(fig);status=dict(status='rendered')
    shared._json(root/'figures/figure_02_examples_source.json',dict(status=status,sources=sources,synthetic=True,additional_biological_measurements=0))
    return status


def _learning_curves(root,ssl_log):
    import matplotlib.pyplot as plt
    rows=[]
    for arm in ARMS:
        for path in sorted((root/'arms'/arm).glob('fold_*/train_log.jsonl')):
            for line in path.read_text().splitlines():
                if not line.strip():continue
                row=json.loads(line)
                if all(key in row for key in ['epoch','loss','encoder_lr','head_lr']):
                    rows.append(dict(arm=arm,fold=path.parent.name,source_sha256=shared._sha(path),
                        **{key:row[key] for key in ['epoch','loss','encoder_lr','head_lr']}))
    frame=pd.DataFrame(rows,columns=['arm','fold','source_sha256','epoch','loss','encoder_lr','head_lr'])
    frame.to_csv(root/'figures/figure_06_supervised_loss_lr_source.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for (arm,fold),part in frame.groupby(['arm','fold']):
        axes[0].plot(part.epoch,part.loss,label=arm+'/'+fold)
        axes[1].plot(part.epoch,part.encoder_lr,label=arm+'/'+fold+' encoder')
        axes[1].plot(part.epoch,part.head_lr,ls='--',label=arm+'/'+fold+' head')
    if rows:axes[0].legend(fontsize=6);axes[1].set_yscale('log')
    else:axes[0].text(.5,.5,'Actual supervised logs missing',ha='center',transform=axes[0].transAxes)
    axes[0].set(xlabel='Epoch',ylabel='Raw weighted MSE / training scale²',title='Encoder tail + head training loss')
    axes[1].set(xlabel='Epoch',ylabel='Learning rate; solid encoder / dashed head',title='Separate parameter-group schedules')
    values=shared._logs(ssl_log,'step','loss') if ssl_log else []
    if values:axes[2].plot(*np.asarray(values).T)
    else:axes[2].text(.5,.5,'Historical SSL log missing',ha='center',transform=axes[2].transAxes)
    axes[2].set(xlabel='Original historical SSL step',ylabel='Latent loss',title='Historical SSL; no new pretraining')
    fig.suptitle('Actual supervised encoder/head dynamics; historical SSL is a separate objective')
    fig.tight_layout()
    for ext in ['svg','png']:fig.savefig(root/f'figures/figure_06_learning_curves.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    return dict(training_type='supervised encoder tail and head',logged_points=len(rows),
        historical_ssl=dict(status='available' if values else 'missing',new_ssl=False,
            interpretation='Historical latent loss does not establish downstream benefit from more SSL.'))


def report(root,folds_csv,draws=10000,seed=42,ssl_log=None,examples=True):
    root=Path(root).resolve();folds_csv=Path(folds_csv).resolve();frame=shared._validate_prepared(folds_csv)
    loaded,arms=_load_folds(root,frame)
    paired=sorted(set(loaded['control'])&set(loaded['shuffle_4']))
    audit=_audit(root,frame,loaded,paired)
    predictions={arm:pd.concat([loaded[arm][f][0] for f in paired]).sort_values('image_path').reset_index(drop=True) for arm in ARMS} if paired else {}
    cohort=frame[frame.fold.isin(paired)]
    reports=root/'reports';reports.mkdir(parents=True,exist_ok=True)
    if paired:
        samples,comparisons=shared.paired_bootstrap(cohort[TARGETS].to_numpy(float),{a:p[PREDICTED].to_numpy(float) for a,p in predictions.items()},cohort.group_id,draws=draws,seed=seed)
        for comparison in comparisons.values():comparison['scope']=LIMITS
    else:samples,comparisons=pd.DataFrame(columns=['draw']),{}
    samples.to_csv(reports/'paired_bootstrap.csv',index=False)
    for arm in ARMS:
        arms[arm]['unpaired_completed_folds']=sorted(set(loaded[arm])-set(paired))
        if paired:arms[arm]['global_weighted_r2_raw']=shared.raw_weighted_r2(predictions[arm][TARGETS],predictions[arm][PREDICTED])
    result=dict(experiment=dict(name='supervised_encoder_tail_online_shuffle_v1',encoder_learning=True,encoder_last_blocks=2,
        final_norm_trainable=True,head_learning=True,shuffle_probability=.2,predictor_used=False,new_ssl=False),
        arms=arms,comparisons=comparisons,protocol_audit=audit,limits=LIMITS,
        cohort=dict(scope='partial development only',authorized_folds=list(FOLDS),completed_paired_folds=paired,
            complete_authorized_folds=paired==list(FOLDS),n_original_images=len(cohort),prepared_original_images=len(frame),
            independent_date_groups=cohort.group_id.nunique(),full_oof=False,training_seeds=1),
        bootstrap=dict(draws=draws,seed=seed),publication_ready=False)
    result['scientific_results']=bool(paired) and not audit['tiny_software_smoke']
    if audit['tiny_software_smoke']:
        result['cohort']['scope']='tiny synthetic software smoke only';result['limits']='Tiny software verification only; no scientific results. '+LIMITS
        for row in comparisons.values():row.update(interpretation='Synthetic software smoke; no scientific inference.',ci_supports_positive=False)
    result['label_sum_audit']=shared._tables(frame,predictions,reports)
    # Reuse only numerical-data plotting helpers, not the frozen protocol validation.
    result['figures'],result['learning_dynamics']=shared._figures(root,frame,predictions,result,False,ssl_log)
    _architecture(root)
    result['learning_dynamics']=_learning_curves(root,ssl_log)
    result['figures']['figure_02_examples']=_examples(root,frame) if examples else dict(status='disabled')
    result['provenance']=dict(analysis_datetime_utc=datetime.now(timezone.utc).isoformat(),report_module_sha256=shared._sha(__file__),
        folds_csv_sha256=shared._sha(folds_csv),upstream=shared._read_json(root/'PROVENANCE.json'),
        checkpoint_inventory=shared._read_json(root/'CHECKPOINT_INVENTORY.json'),
        claim='Encoder/head jointly trained, frozen prefix only; not the preceding frozen-cache experiment.')
    shared._json(reports/'paired_comparisons.json',result)
    shared._json(reports/'analysis_provenance.json',result['provenance'])
    (reports/'interpretation.md').write_text(result['limits']+'\n\nCompleted matched folds: '+str(paired)+'\n'
        +'No claim of complete remote preservation, competition scoring or GitHub weight upload.\n')
    code=reports/'code';code.mkdir(exist_ok=True)
    for name in ['supervised_shuffle.py','supervised_shuffle_report.py','block_shuffle_report.py','block_shuffle.py']:
        source=Path(__file__).parent/name
        if source.is_file():shutil.copyfile(source,code/name)
    with zipfile.ZipFile(root/'paper_results.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            if path.is_file() and not path.is_symlink() and path.suffix in {'.py','.json','.jsonl','.csv','.md','.svg','.png'} and path.stat().st_size<=20*1024**2:
                archive.write(path,str(path.relative_to(root)))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--folds-csv',required=True)
    parser.add_argument('--draws',type=int,default=10000);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--ssl-log');parser.add_argument('--examples',action='store_true')
    args=parser.parse_args();result=report(args.root,args.folds_csv,args.draws,args.seed,args.ssl_log,args.examples)
    print(json.dumps({'cohort':result['cohort'],'comparisons':result['comparisons']},allow_nan=False))


if __name__=='__main__':main()
