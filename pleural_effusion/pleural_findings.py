# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
import argparse
import json
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import generate_binary_structure
# from skimage.feature import structure_tensor, structure_tensor_eigenvalues
import math

def FA_map(e1, e2, e3):
    FA = np.sqrt(1/2)* np.sqrt((e1-e2)**2 + (e2 - e3)**2 + (e3-e1)**2)/np.sqrt(e1**2 + e2**2 + e3**2)
    return FA

def measure_gas(img_raw, img_pleu,img_pneum,hxyz,ghu):
    img_pleu=np.uint8(img_pleu)
    mask=img_pleu>0
    deep = np.int8(np.ceil(8)/hxyz[0])

    struct = generate_binary_structure(3, 26)
    img_pneum = ndimage.binary_dilation(img_pneum>0,structure=struct, iterations=deep*2)

    mask_border_1 = (img_pleu==1) & (ndimage.binary_dilation(img_pleu==2,structure=struct, iterations=deep)>0)
    mask_border_2 = (img_pleu==2) & (ndimage.binary_dilation(img_pleu==1,structure=struct, iterations=deep)>0)

    mask_gas = mask.copy()
    mask_gas[img_raw>ghu]=0
 
    mask_gas[img_pneum>0]=0

    tmp, ncomponents = label(mask_gas>0, struct)
    tmp_ = tmp.copy()
    tmp_[ (mask_border_1==0) | (mask_border_2==0) ] = 0

    mask_gas[np.isin(tmp,np.unique(tmp_))]=0
    Vmicrobubble = (np.sum(mask_gas)  * np.prod(hxyz)) /1000
    Vpneum = ( np.sum(img_pneum>0) * np.prod(hxyz) ) /1000
    Vgas = Vmicrobubble + Vpneum

    return Vgas, Vmicrobubble, Vpneum, mask_gas


def measure_hyper(img_raw, img_pleu, hxyz, hhu):
    mask=img_pleu>0
    mask_hyper = (img_raw<200) & (mask) & (img_raw>hhu)
    mask_hyper = ndimage.morphology.binary_fill_holes(mask_hyper>0)
    struc=np.ones(np.int8(np.ceil((1.5,1.5,1.5)/hxyz)))
    mask_hyper = ndimage.binary_erosion(mask_hyper>0,structure=struc)
    mask_hyper = ndimage.binary_dilation(mask_hyper>0,structure=struc)

    struct = generate_binary_structure(3, 3)
    tmp, ncomponents = label(mask_hyper.astype(np.int16) >0, struct)
    unique, counts = np.unique(tmp, return_counts=True)
    mask_hyper=np.where(np.isin(tmp,unique[counts>2000/np.prod(hxyz)]),tmp,0)

    Vhyper = np.sum(mask_hyper>0) * np.prod(hxyz)/1000
    hyper_rate = np.sum(mask_hyper>0)/np.sum(mask>0)

    if math.isnan(hyper_rate):
        hyper_rate = 0

    return Vhyper, hyper_rate, mask_hyper>0


def splitt_size(mask):
    structure = generate_binary_structure(3, 26)
    tmp, ncomponents = label(mask, structure)
    unique, counts = np.unique(tmp, return_counts=True)


def measure_blood_enhance(img_label,img_hyper,hxyz,thick):
    # detect lesions

    struct = generate_binary_structure(3, 26)
    tmp, ncomponents = label(img_hyper.astype(np.int16) >0, struct)
    unique, counts = np.unique(tmp, return_counts=True)
    ind=np.argsort(counts)
    lesion = dict()
    lesion['volume'] = counts*np.prod(hxyz)
    volume = counts*np.prod(hxyz)

    # seperate in and out sides
    deep = np.int8(np.ceil((thick/2)/hxyz[0]))
    mask_in = ndimage.binary_erosion(img_label==2,structure=struct,iterations=deep)
    mask_lung_dil = ndimage.binary_dilation(img_label==1,structure=struct,iterations=deep)

    mask_out = (img_label==2)
    mask_out[mask_in]=0
    mask_out_lung = np.copy(mask_out)
    mask_out_lung[mask_lung_dil==0]=0
    mask_out_effusion = np.copy(mask_out)
    mask_out_effusion[mask_out_lung>0]=0



    # unique_out=np.unique(tmp[mask_out==1])
    # lesion_out=np.where(np.isin(tmp,unique_out),tmp,0)
    # lesion_in=tmp
    # lesion_in[lesion_out>0]=0

    # unique_out=np.unique(tmp[mask_out==1])
    lesion_out=np.copy(tmp)
    lesion_out[mask_in]=0
    lesion_in=np.copy(tmp)
    lesion_in[mask_out>0]=0

    lesion_out_lung = np.copy(tmp)
    lesion_out_lung[mask_out_effusion>0]=0
    lesion_out_effusion = np.copy(tmp)
    lesion_out_effusion[mask_out_lung>0]=0

    lesion_out_lung_effusion_rate = np.sum(lesion_out_effusion)/np.sum(lesion_out_lung)

    mask_out = (mask_out > 0) | (lesion_out>0 )
    mask_in[mask_out]=0
    out_rate = np.sum(lesion_out>0)/np.sum(mask_out)
    in_rate = np.sum(lesion_in>0)/np.sum(mask_in)
    inout_rate = in_rate/out_rate


    # unique_out=np.unique(tmp[mask_out==1])
    lesion_L=np.where(np.isin(tmp,unique[counts>500/np.prod(hxyz)]),tmp,0)
    lesion_L = lesion_L>0
    lesion_L.astype(float)
    

     
    ## create FA Map 

    # # lesion_single = lesion_out_L==unique_out_L[l]
    # sigma = [3, 3, 3]
    # A_elems = structure_tensor(lesion_L>0, sigma=sigma)
    # e1, e2, e3 = structure_tensor_eigenvalues(A_elems)
    # FA = FA_map(e1,e2,e3)*lesion_L
    # FA = np.nan_to_num(FA)
   

    # ## only large lesion in out

    # unique_out=np.unique(tmp[mask_out==1])
    # lesion_out=np.where(np.isin(tmp,unique_out),tmp,0)
    # lesion_in=np.copy(tmp)
    # lesion_in[lesion_out>0]=0

    # unique_out, counts_out = np.unique(lesion_out, return_counts=True)
    # lesion_out_L = np.where(np.isin(lesion_out,unique_out[counts_out>(1000/np.prod(hxyz))]),lesion_out,0)


    # FA_median = np.median(FA[lesion_out_L>0])

    if math.isnan(out_rate):
        out_rate=0
    
    if math.isnan(in_rate):
        in_rate=0
    
    if math.isnan(inout_rate):
        inout_rate=0

    if math.isnan(lesion_out_lung_effusion_rate):
        lesion_out_lung_effusion_rate=0       


    return out_rate, in_rate, inout_rate, mask_out, lesion_out_lung_effusion_rate


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-raw", dest="raw", required=True)
    parser.add_argument("-label", dest="label", required=True)
    parser.add_argument("-pneum", dest="pneum", default="pneumpath")
    parser.add_argument("-hhu", dest="hhu", default="30")
    parser.add_argument("-ghu", dest="ghu", default="-600")
    parser.add_argument("-thick", dest="thick", default="4")
    parser.add_argument("-json", dest="json", required=True)
    parser.add_argument("-class_dir", dest="class_dir", required=True)

    args = parser.parse_args()      

    if isinstance(args.hhu, str):
        hhu=float(args.hhu)
    else:
        hhu=args.hhu

    if isinstance(args.ghu, str):
        ghu=float(args.ghu)
    else:
        ghu=args.ghu

    if isinstance(args.thick, str):
        thick=float(args.thick)
    else:
        thick=args.thick

    Praw = nib.load(args.raw)
    Plabel = nib.load(args.label)


    img_raw = np.squeeze(Praw.get_fdata())
    img_label = np.squeeze(Plabel.get_fdata())

    if args.pneum in "pneumpath":
        img_pneum = np.uint8(img_label==np.pi)
    else:
        img_pneum = np.squeeze(nib.load(args.pneum).get_fdata())
    
    print('fill holes .........')
    img_pleu = ndimage.binary_dilation(img_label==2,structure=np.ones((3,3,3)))
    img_pleu = ndimage.morphology.binary_fill_holes(img_pleu>0)
    img_pleu = np.int8(ndimage.binary_erosion(img_pleu,structure=np.ones((3,3,3))))
   
    img_label[img_pleu==1]=2

    if os.path.exists(args.json):
        os.system('rm -f {0}'.format(args.json))
    #     with open(args.json) as f:
    #         classes_pred = json.load(f)
    # else:
    classes_pred = dict()

    print('measure gas volume .........')
    hxyz = Plabel.header['pixdim'][1:4]



    classes_pred['gas'], classes_pred['microbubble'], classes_pred['pneum'], mask_gas = measure_gas(img_raw,img_pleu,img_pneum,hxyz, ghu)
    mask_gas[np.uint8(img_label)!=2]=0

    print('detect hyper mask .........')
    classes_pred['hyper'], classes_pred['hyper_rate'],mask_hyper = measure_hyper(img_raw,img_pleu,hxyz, hhu)
    # classes_pred['out_rate'],classes_pred['in_rate'],classes_pred['inout_rate'], mask_out, FA, classes_pred['FA_median'] = measure_blood_enhance(img_pleu,mask_hyper,hxyz)
    
    print('detect blood and enhance .........')
    classes_pred['out_rate'],classes_pred['in_rate'],classes_pred['inout_rate'], mask_out, classes_pred['lesion_out_lung_effusion_rate'] = measure_blood_enhance(img_label,mask_hyper,hxyz,thick)
    classes_pred['iorate_index']=classes_pred['inout_rate']*classes_pred['hyper_rate']




    # structure = generate_binary_structure(3, 26)
    # tmp, ncomponents = label(mask_hyper.astype(np.int16) >0, structure)
    # unique, counts = np.unique(tmp, return_counts=True)
    classes_pred['pleu_volume'] = np.sum(img_pleu>0)*np.prod(hxyz)/1000

    print(classes_pred)

    with open(args.json, 'w') as fp:
        data = json.dump(classes_pred,fp)


    P = nib.Nifti1Image(mask_gas,Plabel.affine,Plabel.header)
    nib.save(P,os.path.join(args.class_dir,'pred_gas.nii.gz'))

    P = nib.Nifti1Image(np.uint8(mask_hyper),Plabel.affine,Plabel.header)
    nib.save(P,os.path.join(args.class_dir,'pred_hyper.nii.gz'))

    P = nib.Nifti1Image(mask_out,Plabel.affine,Plabel.header)
    nib.save(P,os.path.join(args.class_dir,'mask_out.nii.gz'))

    ###### P = nib.Nifti1Image(FA,Plabel.affine,Plabel.header)
    ###### nib.save(P,os.path.join(args.class_dir,'FA.nii.gz'))


if __name__ == "__main__":
    main()
# %%

