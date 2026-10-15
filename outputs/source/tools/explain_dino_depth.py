"""Matched saved-depth CPU FP32 diagnostics; no training or dispatch."""
import json, hashlib, time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from biomass_jepa.dinov3_dense import load_dinov3, DINODensityHead
from biomass_jepa.dino_finetune import configure_backbone, _load_delta, forward_dense
from biomass_jepa.within_image_jigsaw import native_views, preprocess
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/paper_analysis_2026_10_11'; OUT.mkdir(exist_ok=True)
STUDY=ROOT/'outputs/dino_depth_study/watch/results/dino_depth_study'
TARGETS=['Dry_Green_g','Dry_Dead_g','Dry_Clover_g','GDM_g','Dry_Total_g']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def main():
 torch.set_num_threads(3)
 import pandas as pd
 selection=json.loads((ROOT/'outputs/jepa_xai_3/selection.json').read_text())
 metadata=json.loads((ROOT/'outputs/final_dual_study/offline/dinov3_metadata.json').read_text())
 results=[]
 for depth in (2,4,8):
  model,_=load_dinov3(ROOT/'outputs/final_dual_study/offline/dinov3_reference',ROOT/'outputs/final_dual_study/offline/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',metadata)
  frame=pd.read_csv(STUDY/f'depth_{depth}_pooled_predictions.csv')
  for item in tqdm(selection['images'],desc=f'Depth {depth} XAI'):
   start=time.monotonic(); fold=int(item['fold']); path=STUDY/f'fold_{fold}/depth_{depth}/selected.pt'
   metrics=json.loads(path.with_name('metrics.json').read_text())
   assert sha(path)==metrics['selected_file_sha256'], 'Selected checkpoint hash differs from saved metrics'
   ck=torch.load(path,weights_only=True,map_location='cpu',mmap=True)
   assert ck['signature']['fold']==fold and ck['signature']['config']['unfreeze_blocks']==depth
   configure_backbone(model,'F2',depth); _load_delta(model,ck['backbone_delta'])
   scale=ck['head']['scale']; head=DINODensityHead(scale);head.load_state_dict(ck['head'],strict=True)
   model.requires_grad_(False).eval();head.requires_grad_(False).eval()
   image=ROOT/'outputs/jepa_xai_3/images'/Path(item['image_path']).name
   assert sha(image)==item['image_sha256']
   views=torch.stack([preprocess(v,512) for v in native_views(image)])[None].requires_grad_(True)
   features=forward_dense(model,views); pred=head(features)
   grad=torch.autograd.grad(pred[0,4],views)[0].detach()[0]
   heat=(grad*views.detach()[0]).abs().sum(1).numpy()
   grid=heat.reshape(2,8,64,8,64).sum((2,4)); top=np.argsort(grid.ravel())[-4:]
   random=np.random.default_rng(42).choice(128,4,replace=False)
   base=pred.detach()[0].numpy();contrib=head.local_contributions(features.detach())[0].detach().numpy()
   del features,pred
   tests={}
   with torch.no_grad():
    for label,indices in [('top_four',top),('random_four',random)]:
     masked=views.detach().clone()
     for index in indices:
      v,c=divmod(int(index),64);r,c=divmod(c,8);masked[0,v,:,r*64:(r+1)*64,c*64:(c+1)*64]=0
     value=head(forward_dense(model,masked))[0].numpy()
     tests[label]={'cells':indices.tolist(),'prediction_g':value.tolist(),'delta_g':(value-base).tolist()}
   row=frame.loc[frame.image_path==item['image_path']].iloc[0]
   assert int(row.fold)==fold
   saved=row[['pred_'+t for t in TARGETS]].to_numpy(float);truth=row[TARGETS].to_numpy(float)
   name=f'{image.stem}_depth_{depth}'
   np.savez_compressed(OUT/f'{name}_raw.npz',total_gradient=grad.numpy(),total_heatmap=heat,local_contributions=contrib)
   fig,axes=plt.subplots(1,3,figsize=(13,4))
   photo=np.array(Image.open(image).resize((1024,512)))
   axes[0].imshow(photo);axes[0].set_title('Complete original photograph')
   full=np.concatenate(heat,axis=1); vmax=max(np.quantile(full,.99),1e-12)
   axes[1].imshow(photo);axes[1].imshow(np.clip(full/vmax,0,1),cmap='inferno',alpha=.6);axes[1].set_title('Total: |gradient × normalized input|')
   local=np.concatenate(contrib.sum(-1),axis=1)/128
   axes[2].imshow(local,cmap='viridis');axes[2].set_title('Model-derived cell contributions (g)')
   for ax in axes:ax.axis('off')
   fig.suptitle(f'Depth {depth}, fold {fold}: Total truth {truth[4]:.2f}g / CPU {base[4]:.2f}g / saved FP8 {saved[4]:.2f}g')
   fig.tight_layout();fig.savefig(OUT/f'{name}_xai.png',dpi=180);fig.savefig(OUT/f'{name}_xai.pdf');plt.close(fig)
   results.append({'depth':depth,'image_path':item['image_path'],'fold':fold,'truth_g':truth.tolist(),'cpu_fp32_prediction_g':base.tolist(),'saved_fp8_prediction_g':saved.tolist(),'max_precision_difference_g':float(abs(base-saved).max()),'occlusion':tests,'checkpoint_sha256':sha(path),'image_sha256':sha(image),'seconds':time.monotonic()-start})
   (OUT/'DINO_XAI_RESULTS.json').write_text(json.dumps({'targets':TARGETS,'sampling':selection['sampling'],'seed':42,'precision':'CPU FP32, original training FP8/BF16','limitations':'Input sensitivity and mean-RGB occlusion are descriptive, can be out of distribution, and do not explain causal mechanisms or measure local biomass. Heatmaps independently normalized; color intensities are not directly comparable across models.','results':results},indent=2))
  del model
if __name__=='__main__':main()
