from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data
import os
if __name__ == '__main__':
    base = "/data4/yhuang/nnUNet_raw_data_base/nnUNet_raw_data/"
    task_name = "Task666_FLARE2022"

    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTr = join(target_base, "labelsTr")

    image_cases = subfiles(target_imagesTr, suffix='.nii.gz', join=False)
    label_cases = subfiles(target_labelsTr, suffix='.nii.gz', join=False)
    test_cases = subfiles(target_imagesTs, suffix='.nii.gz', join=False)

    json_dict = {}
    json_dict['name'] = "FLARE2022"
    json_dict['description'] = "FLARE2022"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "FLARE2022 data for nnunet"
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
    json_dict['numTraining'] = len(image_cases)
    json_dict['numTest'] = len(test_cases)

    # 每个case都是以 name_identifier_modality.nii.gz的形式出现的
    # 由于FLARE2022是CT单模态的图像, 所以每个case的名字的形式是 FLARE_Tr_0001_0000.nii.gz , modality位置统一是0000
    # dataset.json里面记录的文件名要把modality位置的数据去掉!
    json_dict['training'] = [{'image': "./imagesTr/%s" % label, "label": "./labelsTr/%s" % label} for image, label in zip(image_cases, label_cases)]
    json_dict['test'] = ["./imagesTs/%s" % c.split("_0000")[0] + ".nii.gz" for c in test_cases]

    save_json(json_dict, os.path.join(target_base, "dataset.json"))