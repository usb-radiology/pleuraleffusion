# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Load libraries
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import numpy as np
import json
from functools import reduce
import pickle
import os
import argparse
from scipy import ndimage


source=os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("-ct", dest="ct", required=True)
parser.add_argument("-mask", dest="mask", required=True)
parser.add_argument("-meta", dest="meta", required=True)
parser.add_argument("-lr", dest="lr", required=True)
parser.add_argument("-hyper", dest="hyper", required=True)


args = parser.parse_args()

if os.path.exists(args.meta):
    meta = json.load(open (args.meta))
else:
    meta = {}

P = nib.load(args.ct)
Pmask = nib.load(args.mask)
Plr = nib.load(args.lr)
Phyper = nib.load(args.hyper)

img = np.squeeze(P.get_fdata())
mask = np.squeeze(Pmask.get_fdata())
lr = np.squeeze(Plr.get_fdata())
hyper = np.squeeze(Phyper.get_fdata())


res = np.prod(P.header["pixdim"][1:4])
total = np.sum(mask>0)*res/1000

meta["lung"]={}
meta["lung"]["volume_ml"]=int(np.sum(mask==1)*res/1000)
meta["lung"]["percent_of_total"]=int(np.ceil(meta["lung"]["volume_ml"]/total*100))

meta["pleural_effusion"]={}
meta["pleural_effusion"]["volume_ml"]=int(np.sum( (mask==2) & (lr>0))*res/1000)
meta["pleural_effusion"]["percent_of_total"]=100-meta["lung"]["percent_of_total"]
meta["pleural_effusion"]["left_volume"]=int(np.sum( (mask==2) & (lr ==1) )*res/1000)
meta["pleural_effusion"]["right_volume"]=int(np.sum( (mask==2) & (lr ==2) )*res/1000)

if meta["pleural_effusion"]["volume_ml"]==0:
    meta["pleural_effusion"]["percent_of_left"]=0
    meta["pleural_effusion"]["percent_of_right"]=0
else:
    meta["pleural_effusion"]["percent_of_left"]=int(meta["pleural_effusion"]["left_volume"]/meta["pleural_effusion"]["volume_ml"]*100)
    meta["pleural_effusion"]["percent_of_right"]=int(meta["pleural_effusion"]["right_volume"]/meta["pleural_effusion"]["volume_ml"]*100)

meta["chest"]={}
meta["chest"]["volume_ml"]=int(np.sum(mask>0)*res/1000)
meta["chest"]["percent_of_total"]=100

meta["emphysema"]={}
meta["emphysema"]["volume_ml"]=int(np.sum( (img<-950) & (mask==1) )*res/1000)
meta["emphysema"]["percent_of_lung"]='{:.1f}'.format(meta["emphysema"]["volume_ml"]/meta["lung"]["volume_ml"]*100)

meta["infiltrate"]={}
meta["infiltrate"]["volume_ml"]=int(np.sum( (img>-600) & (img<0) & (mask==1) )*res/1000)
meta["infiltrate"]["percent_of_lung"]=int(meta["infiltrate"]["volume_ml"]/meta["lung"]["volume_ml"]*100)

meta["consolidation"]={}
meta["consolidation"]["volume_ml"]=int(np.sum( (img>-200) & (img<0) & (mask==1) )*res/1000)
meta["consolidation"]["percent_of_lung"]='{:.1f}'.format(meta["consolidation"]["volume_ml"]/meta["lung"]["volume_ml"]*100)
print(meta)

## select the z axis 


com = []
struct = ndimage.morphology.generate_binary_structure(3, 26)

tmp, ncomponents = ndimage.measurements.label(hyper>0, struct)
values, counts = np.unique(tmp, return_counts=True)

idx = np.argsort(-counts)

for l in range(len(counts)-1):
    com.append(np.asarray(ndimage.measurements.center_of_mass(tmp==(idx[l+1]))))

tmp, ncomponents = ndimage.measurements.label(mask==2, struct)

for l in range(ncomponents):
    com.append(np.asarray(ndimage.measurements.center_of_mass(tmp==(l+1))))


if len(com)>5:
    num_lesion = 5
else:    
    num_lesion = len(com)
com = com[0:num_lesion]

if ncomponents>0:

    meta["slice_z"]=np.array2string(np.asarray(com)[:,2].astype(int),precision=2, separator=',',suppress_small=True)
    # meta["slice_x"]=np.array2string(np.asarray(com)[:,0].astype(int),precision=2, separator=',',suppress_small=True)
    # meta["slice_y"]=np.array2string(np.asarray(com)[:,1].astype(int),precision=2, separator=',',suppress_small=True)
else:
    Z_sum=np.sum(np.sum(mask==1, axis=0),axis=0)
    z_idx=np.nonzero(Z_sum)
    Z=np.asarray(list(range(z_idx[0][0],z_idx[0][-1],int((z_idx[0][-1]-z_idx[0][0])/6))))
    meta["slice_z"]=np.array2string(Z[1:-1],precision=2, separator=',',suppress_small=True)


json.dump(meta, open(args.meta, 'w'))


# %%



