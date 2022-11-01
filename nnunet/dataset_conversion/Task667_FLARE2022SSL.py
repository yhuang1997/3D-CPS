from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data
import os
if __name__ == '__main__':
    base = "/data4/FLARE2022"
    task_name = "Task667_FLARE2022CPS"

    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(base, "Training", "FLARE22_LabeledCase50", "images")
    target_labelsTr = join(base, "Training", "FLARE22_LabeledCase50", "labels")
    target_imagesTs = join(base, "ValidationSet")
    target_uimagesTr = join(base, "Training", "FLARE22_UnlabeledCase1-1000")

    image_cases = subfiles(target_imagesTr, suffix='.nii.gz', join=False)
    label_cases = subfiles(target_labelsTr, suffix='.nii.gz', join=False)
    uimage_cases = subfiles(target_uimagesTr, suffix='.nii.gz', join=False)
    test_cases = subfiles(target_imagesTs, suffix='.nii.gz', join=False)

    json_dict = {}
    json_dict['name'] = "FLARE2022SSL"
    json_dict['description'] = "FLARE2022 for ssl learning"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "FLARE2022 ssl data for nnunet"
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": 'Liver',
        "2": 'Right kidney',
        "3": 'Spleen',  # 脾
        "4": 'Pancreas',  # 胰腺
        "5": 'Aorta',  # 主动脉
        "6": 'Inferior Vena Cava',  # 下腔静脉
        "7": 'Right Adrenal Gland',  # 肾上腺
        "8": 'Left Adrenal Gland',
        "9": 'Gallbladder',  # 胆囊
        "10": 'Esophagus',  # 食管
        "11": 'Stomach',
        "12": 'Duodenum',  # 十二指肠
        "13": 'Left kidney'
    }

    # 没有把数据都存在nnUNet_raw_data里面, 把数据集的base写到dataset.json里面
    json_dict['imagesTrBase'] = target_imagesTr
    json_dict['labelsTrBase'] = target_labelsTr
    json_dict['imagesTsBase'] = target_imagesTs
    json_dict['uimagesTrBase'] = target_uimagesTr

    json_dict['numTraining'] = len(image_cases)
    json_dict['numTest'] = len(test_cases)
    json_dict['numUnlabeled'] = len(uimage_cases)

    json_dict['training'] = [{'image':join(target_imagesTr, image.split("_0000")[0] + ".nii.gz"),
                              "label":join(target_labelsTr, label)}
                             for image, label in zip(image_cases, label_cases)]
    json_dict['test'] = [join(target_imagesTs, c.split("_0000")[0] + ".nii.gz") for c in test_cases]
    json_dict['unlabeled'] = [join(target_uimagesTr, c.split("_0000.")[0] + ".nii.gz") for c in uimage_cases]

    save_json(json_dict, os.path.join(target_base, "dataset.json"))