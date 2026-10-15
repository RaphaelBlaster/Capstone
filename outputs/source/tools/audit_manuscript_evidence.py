"""Independent CPU evidence audit; no remote writes or training."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];O=R/'outputs/manuscript_audit_response_2026_10_11';O.mkdir(exist_ok=True)
T=['Dry_Green_g','Dry_Dead_g','Dry_Clover_g','GDM_g','Dry_Total_g'];W=np.array([.1,.1,.1,.2,.5])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def score(y,p):return 1-((y-p)**2*W).sum()/(((y-(y*W).sum()/len(y))**2)*W).sum()
base=R/'outputs/dino_resume_rca/full_launch/results/dino_finetune_full';paths={'F1':base/'F1_pooled_predictions.csv','F2':base/'F2_pooled_predictions.csv'}
for d in [2,4,8]:paths[f'depth{d}']=R/f'outputs/dino_depth_study/watch/results/dino_depth_study/depth_{d}_pooled_predictions.csv'
frames={k:pd.read_csv(p).sort_values('image_path').reset_index(drop=True) for k,p in paths.items()};ref=frames['F1'];Y=ref[T].to_numpy();P={k:f[['pred_'+t for t in T]].to_numpy() for k,f in frames.items()}
for k,f in frames.items():assert not f.image_path.duplicated().any();assert f[['image_path','Sampling_Date','fold']].equals(ref[['image_path','Sampling_Date','fold']]);assert np.array_equal(f[T].to_numpy(),Y)
assert np.array_equal(P['F2'],P['depth2'])
foldpath=R/'outputs/final_dual_study/full_results/final_dual_study/provenance/folds.csv';allrows=pd.read_csv(foldpath);assert len(allrows)==357 and allrows.image_path.nunique()==357;assert allrows.groupby('Sampling_Date')['fold'].nunique().max()==1
joined=ref.merge(allrows[['image_path','Sampling_Date','fold']],on='image_path',suffixes=('','_source'),validate='one_to_one');assert (joined.fold==joined.fold_source).all();assert(joined.Sampling_Date==joined.Sampling_Date_source).all()
metrics={}
for k,p in P.items():
 metrics[k]={'raw_global_weighted_r2':float(score(Y,p)),'per_target':{t:{'mae_g':float(abs(Y[:,i]-p[:,i]).mean()),'rmse_g':float(np.sqrt(((Y[:,i]-p[:,i])**2).mean())),'r2':float(1-((Y[:,i]-p[:,i])**2).sum()/((Y[:,i]-Y[:,i].mean())**2).sum())} for i,t in enumerate(T)}}
groups=[np.where(ref.Sampling_Date.to_numpy()==g)[0] for g in sorted(ref.Sampling_Date.unique())];rng=np.random.default_rng(42);draws=[]
for _ in range(10000):draws.append(np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]))
contrasts={}
for a,b in [('F2','F1'),('depth4','depth2'),('depth8','depth2')]:
 vals=np.array([score(Y[ix],P[a][ix])-score(Y[ix],P[b][ix]) for ix in draws]);contrasts[f'{a}-{b}']={'difference':float(score(Y,P[a])-score(Y,P[b])),'conditional_unadjusted_95pct_interval':np.quantile(vals,[.025,.975]).tolist()}
report={'status':'PASS','images':105,'dates':len(groups),'alignment':'exact image/date/fold/label equality','F2_predictions_equal_depth2':True,'bootstrap_draws':10000,'seed':42,'metrics':metrics,'paired_contrasts':contrasts,'source_hashes':{k:sha(p) for k,p in paths.items()}}
(O/'INDEPENDENT_METRIC_RESCORING.json').write_text(json.dumps(report,indent=2))
rows=[]
for name,f in [('all357',allrows),('heldout105',allrows[allrows.fold.isin([0,1])]),('heldout167',allrows[allrows.fold.isin([0,1,2])])]+[(f'date:{date}',f) for date,f in allrows.groupby('Sampling_Date')]:
 a=f.GDM_g-f.Dry_Green_g-f.Dry_Clover_g;b=f.Dry_Total_g-f.GDM_g-f.Dry_Dead_g
 rows.append({'split':name,'images':len(f),'zero_clover_n':int((f.Dry_Clover_g==0).sum()),'zero_dead_n':int((f.Dry_Dead_g==0).sum()),'GDM_identity_max_abs_g':float(abs(a).max()),'GDM_identity_above_0.01g_n':int((abs(a)>.01).sum()),'Total_identity_max_abs_g':float(abs(b).max()),'Total_identity_above_0.01g_n':int((abs(b)>.01).sum()),'Total_q25_g':float(f.Dry_Total_g.quantile(.25)),'Total_median_g':float(f.Dry_Total_g.median()),'Total_q75_g':float(f.Dry_Total_g.quantile(.75))})
pd.DataFrame(rows).to_csv(O/'BIOLOGICAL_TARGET_AUDIT.csv',index=False);allrows[['image_path','Sampling_Date','fold','image_sha256']+T].to_csv(O/'RECOVERED_DATA_MANIFEST.csv',index=False)
ledger=[]
for k,p in paths.items():ledger.append({'claim':k,'evidence':'recovered original saved predictions','prediction_path':str(p.relative_to(R)),'prediction_sha256':sha(p),'images':105,'folds':'0/1','dates':9,'verdict':'PASS metric/alignment; no new execution'})
ledger.append({'claim':'historical0.83404','evidence':'checkpoint family recovered; original split unavailable in reviewed evidence','prediction_path':'','prediction_sha256':'','images':'unknown for strict historical membership','folds':'unverified','dates':'unverified','verdict':'QUARANTINE; not retroactively verified'})
pd.DataFrame(ledger).to_csv(O/'ARTIFACT_RECOVERY_LEDGER.csv',index=False)
fig,ax=plt.subplots(figsize=(8,3.2));labels=list(contrasts);v=[contrasts[k]['difference'] for k in labels];lo=[contrasts[k]['conditional_unadjusted_95pct_interval'][0] for k in labels];hi=[contrasts[k]['conditional_unadjusted_95pct_interval'][1] for k in labels];ax.errorbar(v,range(3),xerr=[np.array(v)-lo,np.array(hi)-v],fmt='o',capsize=4,color='#376580');ax.axvline(0,color='gray',ls='--');ax.set_yticks(range(3),['Two blocks − head-only','Four blocks − two blocks','Eight blocks − two blocks']);ax.set_xlabel('Paired difference in raw global weighted R²');ax.invert_yaxis();fig.tight_layout();fig.savefig(O/'paired_continuation_effects.pdf',bbox_inches='tight');fig.savefig(O/'paired_continuation_effects.png',dpi=300,bbox_inches='tight');plt.close(fig)
mask=np.ones((16,16));
for rs,cs in [(slice(0,6),slice(0,7)),(slice(0,6),slice(9,16)),(slice(10,16),slice(0,7)),(slice(10,16),slice(9,16))]:mask[rs,cs]=0
assert mask.sum()==88
fig,ax=plt.subplots(figsize=(5,4));ax.imshow(mask,cmap='Blues',vmin=0,vmax=1);ax.set_xticks(np.arange(-.5,16),minor=True);ax.set_yticks(np.arange(-.5,16),minor=True);ax.grid(which='minor',color='white',lw=.6);ax.tick_params(which='both',bottom=False,left=False,labelbottom=False,labelleft=False);ax.set_title('Semantic predictor: 88 visible / 256 patches\nFour hidden corner targets: 42 patches each');fig.tight_layout();fig.savefig(O/'semantic_visibility.pdf',bbox_inches='tight');plt.close(fig)
print(json.dumps({'contrasts':contrasts,'target_audit':rows[:3]},indent=2))
