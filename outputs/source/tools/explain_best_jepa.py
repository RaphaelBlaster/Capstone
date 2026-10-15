"""Descriptive CPU sensitivity for highest retained development context model; no training."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from biomass_jepa.infer import reconstruct_model
from biomass_jepa.data import BiomassImages,TARGETS
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/best_jepa_xai_2026_10_11'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 torch.set_num_threads(3)
 selection=json.loads((ROOT/'outputs/jepa_xai_3/selection.json').read_text())
 ref=json.loads((ROOT/'paper/five_target_independent_selection.json').read_text())['all_results'][0]
 assert ref['name']=='cached_context' and ref['raw_global_weighted_r2']==0.8340391905617833
 manifest=json.loads((OUT/'weights/inference_manifest.json').read_text())
 assert manifest['signature']==ref['signature']
 hashes={n:sha(OUT/'weights'/n) for n in ['jepa_encoder.pt','fold_1_best.pt']}
 for n,v in hashes.items():assert v==manifest['sha256'][n],n
 initial=torch.load(OUT/'weights/jepa_encoder.pt',map_location='cpu',weights_only=True,mmap=True)
 saved=torch.load(OUT/'weights/fold_1_best.pt',map_location='cpu',weights_only=True,mmap=True)
 assert saved['signature']==ref['signature']
 print('Verified development-best .834039 context provenance. Reconstructing actual model.',flush=True)
 model=reconstruct_model(initial,saved).eval().requires_grad_(False)
 del initial,saved
 folds=ROOT/'outputs/final_dual_study/full_results/final_dual_study/provenance/folds.csv'
 # Original historical fold bytes are unavailable (pinned saved source returns404).
 # These are descriptive examples; historical held-out status is UNVERIFIED.
 table=pd.read_csv(folds);results=[]
 for idx,row in enumerate(selection['images']):
  t=time.monotonic();p=ROOT/'outputs/jepa_xai_3/images'/Path(row['image_path']).name
  assert sha(p)==row['image_sha256']
  matching=table[table.image_path==row['image_path']];assert len(matching)==1 and int(matching.iloc[0]['fold'])==1
  frame=pd.DataFrame([{**row,'absolute_path':str(p)}])
  x=BiomassImages(frame,512,train=False,views=2)[0][0][None].requires_grad_(True)
  print(f'[{idx+1}/3] full-photo forward and Total input gradient',flush=True)
  pred=model(x);grad=torch.autograd.grad(pred[0,4],x)[0];base=pred.detach()[0].numpy()
  assert torch.isfinite(pred).all() and torch.isfinite(grad).all(), 'Non-finite prediction or input gradient'
  assert float(grad.detach().abs().sum())>0 and float((grad*x).detach().abs().sum())>0, 'Total input gradient is zero'
  heat=(grad*x).detach().abs().sum(2)[0].numpy()
  # Matched non-overlapping 4x4 cells per full square view, mean-RGB replacement.
  importance=heat.reshape(2,4,128,4,128).sum((2,4)).reshape(-1)
  top=np.argsort(importance)[-3:];random=np.random.default_rng(42+idx).choice(32,3,replace=False)
  pert={}
  with torch.no_grad():
   for name,cells in [('top_three',top),('random_three',random)]:
    z=x.detach().clone()
    for c in cells:
     v,c=divmod(int(c),16);r,c=divmod(c,4);z[0,v,:,r*128:(r+1)*128,c*128:(c+1)*128]=0
    y=model(z)[0].numpy();pert[name]={'cell_ids':cells.tolist(),'prediction_g':y.tolist(),'delta_g':(y-base).tolist()}
   reload=model(x.detach())[0].numpy()
  assert np.array_equal(base,reload),'Same-state CPU repeated prediction changed'
  truth=[float(row[k]) for k in TARGETS]
  results.append({'image':row['image_path'],'checkpoint_fold':1,'historical_heldout_status':'UNVERIFIED','truth_g':truth,'cpu_fp32_prediction_g':base.tolist(),'repeat_max_delta_g':float(abs(base-reload).max()),'occlusion':pert,'seconds':time.monotonic()-t})
  np.savez_compressed(OUT/(p.stem+'_attributions.npz'),total_gradient=grad.detach().numpy(),total_heatmap=heat,cell_importance=importance)
  photo=np.array(Image.open(p).convert('RGB').resize((1024,512)))
  fig,axs=plt.subplots(1,3,figsize=(15,4));axs[0].imshow(photo);axs[0].set_title('Original complete photograph')
  h=np.concatenate(heat,axis=1);axs[1].imshow(photo);axs[1].imshow(h/max(np.quantile(h,.99),1e-12),cmap='inferno',vmin=0,vmax=1,alpha=.65);axs[1].set_title('Total: |gradient × normalized input|')
  axs[2].bar(['Top three','Random three'],[pert[k]['delta_g'][4] for k in ['top_three','random_three']]);axs[2].axhline(0,color='black');axs[2].set_ylabel('Total prediction change (g)');axs[2].set_title('Mean-RGB cell occlusion')
  for a in axs[:2]:a.axis('off')
  fig.suptitle(f'{p.stem} — development-best latent-predictor context, checkpoint fold1; historical held-out status unverified\nTotal truth {truth[4]:.2f}g; CPU FP32 prediction {base[4]:.2f}g')
  fig.tight_layout();fig.savefig(OUT/(p.stem+'_explanation.png'),dpi=180);fig.savefig(OUT/(p.stem+'_explanation.pdf'));plt.close(fig)
  evidence={'selection_reference':'paper/five_target_independent_selection.json: cached_context, .8340391905617833 on105 reused development images','architecture':'Historical DINOv3-L backbone + JEPA-style latent-predictor context readout; not official I-JEPA H14','checkpoint_hashes':hashes,'current_fold_table_sha256':sha(folds),'historical_fold_table_sha256_expected':ref['signature']['folds_sha256'],'historical_heldout_status':'UNVERIFIED: original fold bytes unavailable; saved source/output9 requests404','targets':TARGETS,'results':results,'limitations':'Three fixed random photographs selected from prior outputs; historical held-out status is unverified because the original fold bytes are unavailable. Descriptive local sensitivity, not causal explanation or measured local biomass. CPU FP32 differs from original GPU FP8. Occlusion is out-of-distribution. No full competition ensemble used.'}
  (OUT/'results.json').write_text(json.dumps(evidence,indent=2));print(f'[{idx+1}/3] saved {time.monotonic()-t:.1f}s',flush=True)
 (OUT/'REPORT.md').write_text('# Development-best context sensitivity\n\n'+json.dumps(evidence,indent=2)+'\n')
if __name__=='__main__':main()
