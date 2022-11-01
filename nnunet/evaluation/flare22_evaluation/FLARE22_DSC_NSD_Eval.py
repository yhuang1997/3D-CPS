# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 12:59:48 2022

@author: 12593
"""

import numpy as np
import nibabel as nb
import os
from collections import OrderedDict
import pandas as pd
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
join = os.path.join

infer_suffix = "task666_nnUNetPlansv2.1_GSW_None_UNet_SE_SPPF_nnUNetTrainerV2_GSW_fold['all']_3d_fullres_normal"

seg_path = os.path.join('/public/datasets/yhuang/nnunet_workdir/inference_segmentations/', infer_suffix)
gt_path = '/public/datasets/FLARE2022/ReleaseValGT-20cases'
save_name = 'DSC_NSD.csv'

filenames = os.listdir(seg_path)
filenames = [x for x in filenames if x.endswith('.nii.gz')]
filenames.sort()

seg_metrics = OrderedDict()
seg_metrics['Name'] = list()
label_tolerance = OrderedDict({'Liver': 5, 'RK':3, 'Spleen':3, 'Pancreas':5, 
                   'Aorta': 2, 'IVC':2, 'RAG':2, 'LAG':2, 'Gallbladder': 2,
                   'Esophagus':3, 'Stomach': 5, 'Duodenum': 7, 'LK':3})
for organ in label_tolerance.keys():
    seg_metrics['{}_DSC'.format(organ)] = list()
for organ in label_tolerance.keys():
    seg_metrics['{}_NSD'.format(organ)] = list()

def find_lower_upper_zbound(organ_mask):
    """
    Parameters
    ----------
    seg : TYPE
        DESCRIPTION.

    Returns
    -------
    z_lower: lower bound in z axis: int
    z_upper: upper bound in z axis: int

    """
    organ_mask = np.uint8(organ_mask)
    assert np.max(organ_mask) ==1, print('mask label error!')
    z_index = np.where(organ_mask>0)[2]
    z_lower = np.min(z_index)
    z_upper = np.max(z_index)
    
    return z_lower, z_upper



for name in filenames:
    # load grond truth and segmentation
    if os.path.exists(join(gt_path, name)):
        # gt只有20个, 但seg有50个
        seg_metrics['Name'].append(name)
        gt_nii = nb.load(join(gt_path, name))
    else:
        continue
    case_spacing = gt_nii.header.get_zooms()
    gt_data = np.uint8(gt_nii.get_fdata())
    seg_data = np.uint8(nb.load(join(seg_path, name)).get_fdata())

    for i, organ in enumerate(label_tolerance.keys(),1):
        if np.sum(gt_data==i)==0 and np.sum(seg_data==i)==0:
            DSC_i = 1
            NSD_i = 1
        elif np.sum(gt_data==i)==0 and np.sum(seg_data==i)>0:
            DSC_i = 0
            NSD_i = 0
        else:
            if i==5 or i==6 or i==10: # for Aorta, IVC, and Esophagus, only evaluate the labelled slices in ground truth
                z_lower, z_upper = find_lower_upper_zbound(gt_data==i)
                organ_i_gt, organ_i_seg = gt_data[:,:,z_lower:z_upper]==i, seg_data[:,:,z_lower:z_upper]==i
            else:
                organ_i_gt, organ_i_seg = gt_data==i, seg_data==i
            surface_distances = compute_surface_distances(organ_i_gt, organ_i_seg, case_spacing)
            DSC_i = compute_dice_coefficient(organ_i_gt, organ_i_seg)
            NSD_i = compute_surface_dice_at_tolerance(surface_distances, label_tolerance[organ])
        seg_metrics['{}_DSC'.format(organ)].append(round(DSC_i, 4))
        seg_metrics['{}_NSD'.format(organ)].append(round(NSD_i, 4))
        print(name, organ, round(DSC_i,4), 'tol:', label_tolerance[organ], round(NSD_i,4))

dataframe = pd.DataFrame(seg_metrics)
col_means = dataframe.mean(axis=0)
dataframe.loc[len(dataframe.index)] = col_means  # 最后一行+mean

dsc_mean = col_means[0:13].mean()
nsd_mean = col_means[13:].mean()
dsc_std = col_means[0:13].std()
nsd_std = col_means[13:].std()

overview_ss =  pd.Series([dsc_mean, nsd_mean, dsc_std, nsd_std], index=['dsc_mean', 'nsd_mean', 'dsc_std', 'nsd_std'])

dataframe.to_csv(join(seg_path, save_name), index=False)
overview_ss.to_csv(join(seg_path, 'overview_' + save_name), index=False)








