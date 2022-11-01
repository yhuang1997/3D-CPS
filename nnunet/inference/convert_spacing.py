import os
from os.path import join, split, exists
import nibabel as nib
from glob import glob
from batchgenerators.utilities.file_and_folder_operations import *

def convert_target_spacing(input_path, output_path, target_spacing=[1,1,1.5]):
    target_spacing_z = target_spacing[2]
    file_lists = sorted(glob(join(input_path, '*.nii.gz')))
    spacings = []
    if not exists(output_path):
        os.makedirs(output_path)
    for f in file_lists:
        img = nib.load(f)
        num_slices = img.shape[2]

        # slices多的进一步降spacing
        if num_slices > 600:
            reduce_factor = max(0.8, 600/num_slices)
            target_spacing[2] = target_spacing_z * reduce_factor
        # slices少的保持spacing,不做修改
        elif num_slices < 150:
            target_spacing[2] = img._affine[2][2]
        else:
            target_spacing[2] = target_spacing_z
        original_spacing = [img._affine[0][0], img._affine[1][1], img._affine[2][2]]
        print(img._affine)
        img._affine[0][0] = target_spacing[0]
        img._affine[1][1] = target_spacing[1]
        img._affine[2][2] = target_spacing[2]
        spacings.append(original_spacing)
        print(img._affine)
        _, filename = split(f)
        nib.save(img, join(output_path, filename))
    save_pickle(spacings, join(output_path, 'spacing.pkl'))

def convert_original_spacing(input_path, output_path, spacing_pkl):
    file_lists = sorted(glob(join(input_path, '*.nii.gz')))
    if not exists(output_path):
        os.makedirs(output_path)
    original_spacing = load_pickle(spacing_pkl)
    assert len(file_lists) == len(original_spacing)
    for f, s in zip(file_lists, original_spacing):
        img = nib.load(f)
        img._affine[0][0] = s[0]
        img._affine[1][1] = s[1]
        img._affine[2][2] = s[2]
        _, filename = split(f)
        nib.save(img, join(output_path, filename))



if __name__ == '__main__':
    # input_path = "/public/datasets/yhuang/Dataset/ValidationSet-20cases/"
    input_path = "/data4/FLARE2022/ValidationSet"
    # output_path = "/public/datasets/yhuang/Dataset/ValidationSet-20cases-preprocessed/"
    output_path = "/public/datasets/yhuang/Dataset/ValidationSet-preprocessed/"

    # output_path = "/public/datasets/yhuang/Dataset/ValidationSet-20cases-preprocessed-back/"
    convert_target_spacing(input_path, output_path)
