"""Render publication panels from preserved metrics and prediction arrays; no training."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
R=Path(__file__).resolve().parents[1]
O=R/'paper/scientific_reports_2026_10_11/revision_2/figures';O.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def save(fig,name):
 fig.savefig(O/f'{name}.pdf',bbox_inches='tight');fig.savefig(O/f'{name}.png',dpi=220,bbox_inches='tight');plt.close(fig)
f,ax=plt.subplots(figsize=(9,4.1));ax.axis('off')
boxes=[(.02,.60,.43,.29,'Official I-JEPA ViT-H/14 · two 224 px views\nOriginal / grass-adapted encoder / predictor\nFrozen readouts and matched predictor specialization'),(.55,.60,.43,.29,'Official DINOv3 ViT-B/16 · two 512 px views\n32×32 patch tokens → 8×8 equal-area cells\nGlobal / spatial / cross-scale regression'),(.02,.15,.43,.29,'105 photographs · folds 0/1 · nine dates\nSeparate frozen-shuffle study: 167 photographs\nNo shuffle / 4×4 / 10×10; folds 0/1/2'),(.55,.15,.43,.29,'Frozen D2 head → matched continuation\nLast 2 / 4 / 8 blocks + normalization\n20 epochs · fixed pretrained anchoring')]
for x,y,w,h,t in boxes:
 ax.add_patch(plt.Rectangle((x,y),w,h,facecolor='#eff5f8',edgecolor='#376580',lw=1.2));ax.text(x+w/2,y+h/2,t,ha='center',va='center',linespacing=1.7)
ax.annotate('',xy=(.765,.44),xytext=(.765,.60),arrowprops={'arrowstyle':'->','color':'#376580'});ax.text(.5,.035,'Photograph-level labels · original validation images · pooled weighted R²\nPaired date-group uncertainty · preserved checkpoints · descriptive sensitivity analysis',ha='center',va='center')
save(f,'study_overview')
f,aa=plt.subplots(2,2,figsize=(9,6.4))
series=[('a  Official I-JEPA readouts · n=105',['Original','Adapted encoder','Predictor'],[.5464863,.586007,.5721636]),('b  Predictor specialization · n=105',['Unshuffled','Shuffled'],[.5489118,.556713]),('c  Frozen permutation augmentation · n=167',['Control','4×4','10×10'],[.6163042,.5835414,.5654594]),('d  Frozen DINOv3 readouts · n=105',['Global (D0)','Spatial (D1)','Cross-scale (D2)'],[.7659125,.7798814,.7800651])]
for a,(title,l,v) in zip(aa.flat,series):
 a.bar(l,v,color=['#577f9c','#2e8b83','#b79056'][:len(v)],width=.58);a.set_ylim(0,.9);a.set_ylabel('Raw global weighted R²');a.set_title(title,loc='left');a.tick_params(axis='x',labelsize=8)
 for i,z in enumerate(v):a.text(i,z+.025,f'{z:.4f}',ha='center')
f.tight_layout();save(f,'model_comparisons')
f,aa=plt.subplots(2,3,figsize=(9,5.7),sharex='row',sharey='row')
for col,d in enumerate([2,4,8]):
 p=R/f'outputs/dino_depth_study/watch/results/dino_depth_study/depth_{d}_pooled_predictions.csv';df=pd.read_csv(p);y=df.Dry_Total_g.to_numpy();z=df.pred_Dry_Total_g.to_numpy()
 aa[0,col].scatter(y,z,s=14,alpha=.65,color='#376580');aa[0,col].plot([0,190],[0,190],ls='--',color='gray');aa[0,col].set_title(f'{d} trainable blocks');aa[0,col].set_xlabel('Measured Total (g)');aa[0,col].set_xlim(0,190);aa[0,col].set_ylim(0,190)
 aa[1,col].scatter(y,z-y,s=14,alpha=.65,color='#2e8b83');aa[1,col].axhline(0,ls='--',color='gray');aa[1,col].set_xlabel('Measured Total (g)');aa[1,col].set_xlim(0,190)
aa[0,0].set_ylabel('Predicted Total (g)');aa[1,0].set_ylabel('Prediction − measured (g)');f.tight_layout();save(f,'depth_error_analysis')
for image in ['ID567744300','ID1211362607','ID1103883611']:
 f,aa=plt.subplots(3,3,figsize=(9,6.7));original=np.asarray(Image.open(R/f'outputs/jepa_xai_3/images/{image}.jpg'))
 for col,d in enumerate([2,4,8]):
  raw=np.load(R/f'outputs/paper_analysis_2026_10_11/{image}_depth_{d}_raw.npz');heat=raw['total_heatmap'];local=raw['local_contributions'].sum(-1)/128
  aa[0,col].imshow(original);aa[0,col].set_title(f'{d} trainable blocks');aa[0,col].axis('off')
  for row,array,cmap,label in [(1,heat,'magma','Normalized |input × gradient|'),(2,local,'viridis','Model-derived contribution (g)')]:
   arr=np.concatenate(list(array),axis=1);arr=arr/max(float(np.quantile(arr,.99)),1e-12) if row==1 else arr;im=aa[row,col].imshow(arr,cmap=cmap,vmin=0,vmax=1 if row==1 else None);aa[row,col].set_xticks([]);aa[row,col].set_yticks([]);aa[row,col].set_xlabel(label,fontsize=8);f.colorbar(im,ax=aa[row,col],fraction=.025,pad=.02)
 f.tight_layout();save(f,'matched_xai' if image=='ID567744300' else f'xai_{image}')
