# 3D-CPS
Codes for the paper: ["3D Cross Pseudo Supervision (3D-CPS): A semi-supervised nnU-Net architecture for abdominal organ segmentation"](https://arxiv.org/abs/2209.08939) by Yongzhi Huang*, Hanwen Zhang*, Yan Yan and Haseeb Hassan.
# Dataset and Challenge

[Fast and Low-resource semi-supervised Abdominal oRgan sEgmentation in CT (FLARE 2022)](https://flare22.grand-challenge.org/)

# Usage

Our framework is based on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), so we **strongly recommend you have a look at the repository of nnU-Net before starting with our code.**

## Dataset Conversion

nnU-Net expects datasets in a structured format like the data structure of the [Medical Segmentation Decthlon](http://medicaldecathlon.com/). The first step is to build `dataset.json` using `dataset_convErsion.TaskXXX_TASKNAME.py`.

## Experiment planning and preprocessing

We follow the network structure and other hyper-parameter settings automatically generated by nnU-Net, so the default `ExperimentPlanner` is enough.

`python nnUNet_plan_and_preprocess.py -t TASK_ID --ssl`

If you want to change any properties related to models or training hyper-parameters, you can inherit and override `Class ExperimentPlanner3D_v21/ExperimentPlanner2D_v21`, and modify the `Class Trainer` (In our work, it's `nnUNetTrainerV2_SSL`) accordingly. For more details, please refer to [this](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/extending_nnunet.md) guide in the nnU-Net repository.

`python nnUNet_plan_and_preprocess.py -t TASK_ID -p YOUR_EXP_PLANNER --ssl`

## Model training

- 2D version: 

  `python run_training.py 2d nnUNetTrainerV2_SSL TASK_ID FOLD`

- 3D version:

  `python run_training.py 3d_fullres nnUNetTrainerV2_SSL TASKID FOLD`

## Inference

- 2D version

  `python predict_simple.py -i INPUT_DIR -o OUTPUT_DIR -f FOLD -t TASK_ID -m 3d_fullres -tr nnUNetTrainerV2_SSL -p nnUNetPlansv2.1 -chk model_best`

- 3D version:

  `python predict_simple.py -i INPUT_DIR -o OUTPUT_DIR -f FOLD -t TASK_ID -m 2d -tr nnUNetTrainerV2_SSL -p nnUNetPlansv2.1 -chk model_best`

# Reference

- [[CVPR 2021\] Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision](https://arxiv.org/abs/2106.01226)

- [nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation](https://www.nature.com/articles/s41592-020-01008-z)