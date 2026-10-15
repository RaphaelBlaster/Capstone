"""Expand saved DINOv3 attribution/prediction figures without retraining."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];A=R/'outputs/paper_analysis_2026_10_11';O=R/'outputs/dino_xai_expanded_2026_10_11';O.mkdir(exist_ok=True)
F=O/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
results=json.loads((A/'DINO_XAI_RESULTS.json').read_text());checks=[]
def save(fig,name):
 fig.savefig(F/f'{name}.pdf',bbox_inches='tight');fig.savefig(F/f'{name}.png',dpi=230,bbox_inches='tight');plt.close(fig)
for image in ['ID567744300','ID1211362607','ID1103883611']:
 photo=np.asarray(Image.open(R/f'outputs/jepa_xai_3/images/{image}.jpg').resize((1024,512)))
 fig,ax=plt.subplots(3,3,figsize=(14,8.5))
 for row,d in enumerate([2,4,8]):
  item=next(x for x in results['results'] if x['depth']==d and image in x['image_path']);raw=A/f'{image}_depth_{d}_raw.npz';z=np.load(raw);h=np.concatenate(z['total_heatmap'],axis=1);h=np.clip(h/max(float(np.quantile(h,.99)),1e-12),0,1);cells=np.concatenate(z['local_contributions'].sum(-1),axis=1)/128
  assert np.isfinite(h).all() and h.max()>0;assert abs(cells.sum()-item['cpu_fp32_prediction_g'][4])<2e-5
  ax[row,0].imshow(photo);ax[row,0].set_title(f'{d} blocks · measured Total {item["truth_g"][4]:.2f} g\nCPU prediction {item["cpu_fp32_prediction_g"][4]:.2f} g')
  ax[row,1].imshow(photo);im=ax[row,1].imshow(h,cmap='inferno',vmin=0,vmax=1,alpha=.55);ax[row,1].set_title('Total input-sensitivity overlay');fig.colorbar(im,ax=ax[row,1],fraction=.025,pad=.01)
  im=ax[row,2].imshow(cells,cmap='viridis',vmin=0);ax[row,2].set_title('Model-derived contribution proxies (g)');fig.colorbar(im,ax=ax[row,2],fraction=.025,pad=.01)
  for a in ax[row]:a.set_xticks([]);a.set_yticks([])
  checks.append({'image':image,'depth':d,'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'finite_nonzero_heatmap':True,'contribution_sum_error_g':float(cells.sum()-item['cpu_fp32_prediction_g'][4])})
 fig.suptitle(f'{image}: complete views, no image selection by prediction error',fontsize=13);fig.tight_layout();save(fig,f'{image}_depth_comparison_overlays')
fig,ax=plt.subplots(1,3,figsize=(13,3.7),sharey=True)
for a,d in zip(ax,[2,4,8]):
 items=[x for x in results['results'] if x['depth']==d];x=np.arange(3)
 a.bar(x-.18,[i['occlusion']['top_four']['delta_g'][4] for i in items],width=.36,label='Top four gradient cells',color='#bc6547');a.bar(x+.18,[i['occlusion']['random_four']['delta_g'][4] for i in items],width=.36,label='Four random cells',color='#376580');a.axhline(0,color='gray',lw=.8);a.set_xticks(x,['Image 1','Image 2','Image 3']);a.set_title(f'{d} trainable blocks')
ax[0].set_ylabel('Occluded − original Total prediction (g)');ax[2].legend(fontsize=8);fig.tight_layout();save(fig,'occlusion_comparison')
targets=['Dry_Green_g','Dry_Dead_g','Dry_Clover_g','GDM_g','Dry_Total_g'];colors=['#376580','#388f7b','#a77840'];frames=[]
for d in [2,4,8]:
 df=pd.read_csv(A/f'source_data/depth_{d}_pooled_predictions.csv');frames.append(df)
 fig,ax=plt.subplots(2,3,figsize=(12,7))
 for a,t in zip(ax.flat,targets):
  y=df[t];p=df['pred_'+t];limit=max(y.max(),p.max())*1.05;a.scatter(y,p,s=19,alpha=.65,color=colors[[2,4,8].index(d)]);a.plot([0,limit],[0,limit],ls='--',color='gray');a.set(xlabel='Measured (g)',ylabel='Predicted (g)',title=t.replace('Dry_','').replace('_g',''),xlim=(0,limit),ylim=(0,limit))
 ax.flat[-1].axis('off');fig.suptitle(f'DINOv3: {d} trainable blocks · 105 reused development photographs');fig.tight_layout();save(fig,f'depth_{d}_all_target_scatter')
fig,ax=plt.subplots(1,2,figsize=(10,3.7));metrics=pd.read_csv(A/'depth_per_target_metrics.csv');print(metrics.columns.tolist())
for i,d in enumerate([2,4,8]):
 df=frames[i];mae=[np.mean(abs(df['pred_'+t]-df[t])) for t in targets];rmse=[np.sqrt(np.mean((df['pred_'+t]-df[t])**2)) for t in targets];ax[0].plot(range(5),mae,'o-',label=f'{d} blocks',color=colors[i]);ax[1].plot(range(5),rmse,'o-',color=colors[i])
for a,label in zip(ax,['MAE (g)','RMSE (g)']):a.set_xticks(range(5),['Green','Dead','Clover','GDM','Total']);a.set_ylabel(label)
ax[0].legend();fig.tight_layout();save(fig,'all_target_errors')
(O/'CHECKS.json').write_text(json.dumps({'depths_available':[2,4,8],'depth_6_checkpoint_found':False,'cases':checks,'precision':'Existing CPU FP32 attributions; original saved predictions FP8/BF16','display':'99th-percentile normalization separately per heatmap, 0-1 clipped','limitations':'Input sensitivity and artificial occlusion, not causal explanation or measured spatial biomass.'},indent=2))
