"""Ten native within-parent permutations. No independent biological samples."""
import hashlib
import random
import numpy as np
from PIL import Image,ImageOps
import torch

def permutation_bank(image_sha256,view,master_seed=42):
 orders=[]
 for i in range(10):
  seed=int.from_bytes(hashlib.sha256(f'{image_sha256}|{view}|{i}|{master_seed}'.encode()).digest(),'big')
  rng=random.Random(seed)
  while True:
   order=list(range(100));rng.shuffle(order)
   if order!=list(range(100)) and order not in orders:break
  orders.append(order)
 return orders

def apply_permutation(tile,order):
 if tile.size!=(1000,1000) or tile.mode!='RGB' or sorted(order)!=list(range(100)):raise ValueError('Native1000RGB bijective100-cell permutation required.')
 out=Image.new('RGB',tile.size)
 for dest,src in enumerate(order):
  x,y=src%10*100,src//10*100;out.paste(tile.crop((x,y,x+100,y+100)),(dest%10*100,dest//10*100))
 return out

def native_views(path):
 with Image.open(path) as im:im=ImageOps.exif_transpose(im).convert('RGB')
 if im.size!=(2000,1000):raise ValueError('Expected full2000x1000 photograph after EXIF.')
 return [im.crop((0,0,1000,1000)),im.crop((1000,0,2000,1000))]

def preprocess(tile,size):
 a=np.array(tile.resize((size,size),Image.Resampling.BICUBIC),copy=True)
 x=torch.from_numpy(a).permute(2,0,1).float()/255
 return (x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
