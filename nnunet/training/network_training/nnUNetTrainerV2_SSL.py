#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from _warnings import warn
from collections import OrderedDict
from typing import Tuple
from multiprocessing import Pool
from time import time,sleep
import shutil
from copy import deepcopy

from tqdm import trange
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
import torch.backends.cudnn as cudnn
from nnunet.configuration import default_num_threads
from nnunet.evaluation.evaluator import aggregate_scores
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from nnunet.postprocessing.connected_components import determine_postprocessing
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation, get_moreDA_augmentation_ssl
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoader3D, DataLoader2D, unpack_dataset
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *


class nnUNetTrainerV2_SSL(nnUNetTrainerV2):
    """
    Trainer for SSL
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # new attributes has designed in nnUNetTrainerV2
        # self.max_num_epochs = 1000
        # self.initial_lr = 1e-2
        # self.deep_supervision_scales = None
        # self.ds_loss_weights = None
        # self.pin_memory = True
        policies = {'CPS'}
        co_training = {'CPS'}

        # 如果想给这个trainer.init增加新的参数, 注意要把这些参数加到self.init_args!
        # 看nnUNetTrainer的__init__
        # 建议的写法
        # all_args = []
        # for arg in args:
        #     all_args.append(args)

        self.ssl_loss_weight = 0.5
        self.ssl_policy = 'CPS'
        self.batch_ratio = 1
        if self.ssl_policy in co_training:
            self.co_training = True
        elif self.ssl_policy in policies:
            self.co_training = False
        else:
            raise  RuntimeError("policy {} is not defined in policies {}".format(self.ssl_policy, policies))

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            dl_tr_labeled = DataLoader3D(self.dataset_tr_labeled, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_tr_unlabeled = DataLoader3D(self.dataset_tr_unlabeled, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                         False, oversample_foreground_percent=self.oversample_foreground_percent,
                                         pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        else:
            dl_tr_labeled = DataLoader2D(self.dataset_tr_labeled, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                         oversample_foreground_percent=self.oversample_foreground_percent,
                                         pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_tr_unlabeled = DataLoader2D(self.dataset_tr_unlabeled, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                           oversample_foreground_percent=self.oversample_foreground_percent,
                                           pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
            dl_val = DataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        return dl_tr_labeled, dl_tr_unlabeled, dl_val

    def process_plans(self, plans):
        super().process_plans(plans)
        # process new params for trainer
        warm_up_lambda = plans.get('warm_up_lambda', None)
        if warm_up_lambda is not None:
            self.warm_up_lambda = warm_up_lambda
        else:
            self.warm_up_lambda = False

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)  # set attributes from plans

            # TODO: augmentation params 需要支持 labeled 和 unlabeled
            self.setup_DA_params()  # set data augmentation params

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)  # 有net_numpool个output!
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                # TODO: self.dl_tr -> labeled + unlabeled
                self.dl_tr_labeled, self.dl_tr_unlabeled, self.dl_val = self.get_basic_generators()

                # 这一步将npz -> npy, 非常耗时, 1000unlabeled + 50 labeled 大概需要570GB的存储空间
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_labeled_gen, self.tr_unlabeled_gen, self.val_gen = get_moreDA_augmentation_ssl(
                    self.dl_tr_labeled, self.dl_tr_unlabeled, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr_labeled.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr_unlabeled.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network[0], (SegmentationNetwork, nn.DataParallel))
            assert isinstance(self.network[1], (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        # TODO:
        #  co-training
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}

        # FIXME: initialization method?
        network_1 = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)

        network_2 = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        # network_2 = deepcopy(network_1)
        self.network = (network_1, network_2)

        if torch.cuda.is_available():
            for net in self.network:
                net.cuda()
        for net in self.network:
            net.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        optimizer_1 = torch.optim.SGD(self.network[0].parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        optimizer_2 = torch.optim.SGD(self.network[1].parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)

        self.optimizer = (optimizer_1, optimizer_2)
        self.lr_scheduler = (None, None)

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, train=True):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        # assert not (isinstance(data_generator, tuple) or do_backprop), \
        #     "Data generator should be tuple type in ssl training" \
        #     "including labeled & unlabeled data generator."
        if train:
            labeled_data_dict = next(data_generator[0])
            unlabeled_data_dict = next(data_generator[1])
            labeled_data, unlabeled_data = labeled_data_dict['data'], unlabeled_data_dict['data']
            target = labeled_data_dict['target']
            labeled_data = maybe_to_torch(labeled_data)
            unlabeled_data = maybe_to_torch(unlabeled_data)
            target = maybe_to_torch(target)

            if torch.cuda.is_available():
                labeled_data = self.maybe_apply_for_tuple(to_cuda, labeled_data)
                unlabeled_data = self.maybe_apply_for_tuple(to_cuda, unlabeled_data)
                target = self.maybe_apply_for_tuple(to_cuda, target)

        else:
            labeled_data_dict = next(data_generator)
            labeled_data= labeled_data_dict['data']
            target = labeled_data_dict['target']
            labeled_data = maybe_to_torch(labeled_data)
            target = maybe_to_torch(target)

            if torch.cuda.is_available():
                labeled_data = self.maybe_apply_for_tuple(to_cuda, labeled_data)
                target = self.maybe_apply_for_tuple(to_cuda, target)

        self.optimizer[0].zero_grad()
        self.optimizer[1].zero_grad()

        # outputs shape: list of [B, C, H, W]
        if self.fp16:
            with autocast():
                if train:
                    pred_sup_1 = self.network[0](labeled_data)
                    pred_sup_2 = self.network[1](labeled_data)

                    pred_unsup_1 = self.network[0](unlabeled_data)
                    pred_unsup_2 = self.network[1](unlabeled_data)

                    pred_1 = [torch.cat([sup, unsup], dim=0) for sup, unsup in zip(pred_sup_1, pred_unsup_1)]
                    pred_2 = [torch.cat([sup, unsup], dim=0) for sup, unsup in zip(pred_sup_2, pred_unsup_2)]

                    label_1 = [torch.argmax(p1, dim=1).unsqueeze(1) for p1 in pred_1]
                    label_2 = [torch.argmax(p2, dim=1).unsqueeze(1) for p2 in pred_2]

                    del labeled_data, unlabeled_data

                    l_cps = (self.loss(pred_1, label_2) + self.loss(pred_2, label_1)) * self.ssl_loss_weight
                    l_1 = self.loss(pred_sup_1, target)
                    l_2 = self.loss(pred_sup_2, target)

                    l = l_1 + l_2 + l_cps
                    print("l_cps:{}, l_1:{}, l_2:{}, l:{}".format(l_cps, l_1, l_2, l))
                else:
                    # 验证只拿第一个model
                    pred_sup = self.network[0](labeled_data)
                    l = self.loss(pred_sup, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer[0])
                self.amp_grad_scaler.unscale_(self.optimizer[1])
                torch.nn.utils.clip_grad_norm_(self.network[0].parameters(), 12)
                torch.nn.utils.clip_grad_norm_(self.network[1].parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer[0])
                self.amp_grad_scaler.step(self.optimizer[1])
                self.amp_grad_scaler.update()
        else:
            if train:
                pred_sup_1 = self.network[0](labeled_data)
                pred_sup_2 = self.network[1](labeled_data)

                pred_unsup_1 = self.network[0](unlabeled_data)
                pred_unsup_2 = self.network[1](unlabeled_data)

                pred_1 = [torch.cat([sup, unsup], dim=0) for sup, unsup in zip(pred_sup_1, pred_unsup_1)]
                pred_2 = [torch.cat([sup, unsup], dim=0) for sup, unsup in zip(pred_sup_2, pred_unsup_2)]

                label_1 = [torch.argmax(p1, dim=1).unsqueeze(1) for p1 in pred_1]
                label_2 = [torch.argmax(p2, dim=1).unsqueeze(1) for p2 in pred_2]

                del labeled_data, unlabeled_data

                l_cps = self.loss(pred_1, label_2) + self.loss(pred_2, label_1)
                l_1 = self.loss(pred_sup_1, target)
                l_2 = self.loss(pred_sup_2, target)

                l = l_1 + l_2 + l_cps * self.ssl_loss_weight
            else:
                # 验证只拿第一个model
                pred_sup = self.network[0](labeled_data)
                l = self.loss(pred_sup, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network[0].parameters(), 12)
                torch.nn.utils.clip_grad_norm_(self.network[1].parameters(), 12)
                self.optimizer[0].step()
                self.optimizer[1].step()

        if run_online_evaluation:
            self.run_online_evaluation(pred_sup, target)

        del target

        return l.detach().cpu().numpy()

    def maybe_apply_for_tuple(self, f, data, *f_args, **f_kwargs):
        """
        要以元组的形式存储多个network，多个optimizer等
        每次对元组or单个对象执行一些function的时候很麻烦，用这个函数看起来方便点
        :param f:
        :param data:
        :param f_args:
        :param f_kwargs:
        :return:
        """
        if isinstance(data, (tuple, list)):
            ret = []
            for d in data:
                ret.append(f(d, *f_args, **f_kwargs))
            return tuple(ret)
        else:
            return f(data, *f_args, **f_kwargs)

    def do_split(self):
        """
        Overwritten by yhuang, it's compatible with labeled + unlabeled dataset
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                labeled_keys_sorted, unlabeled_keys_sorted = self._labeled_unlabeled_filter(all_keys_sorted)
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(labeled_keys_sorted)):
                    train_keys = np.array(labeled_keys_sorted)[train_idx]
                    test_keys = np.array(labeled_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = np.concatenate((train_keys, unlabeled_keys_sorted))
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()

        tr_labeled_keys, tr_unlabeled_keys = self._labeled_unlabeled_filter(tr_keys)
        val_keys, _ = self._labeled_unlabeled_filter(val_keys)  # unlabeled data去掉
        self.dataset_tr_labeled = OrderedDict()
        for i in tr_labeled_keys:
            self.dataset_tr_labeled[i] = self.dataset[i]

        self.dataset_tr_unlabeled = OrderedDict()
        for i in tr_unlabeled_keys:
            self.dataset_tr_unlabeled[i] = self.dataset[i]

        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def _labeled_unlabeled_filter(self, sorted_datasets):
        # 只针对FLARE2022 Dataset, 有2000个unlabeled和50个labeled, 逻辑有问题但先这样～
        for i, identifier in enumerate(sorted_datasets):
            if not identifier[:4] == 'Case':
                break
        return sorted_datasets[i:], sorted_datasets[:i]

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer[0].param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.optimizer[1].param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer[0].param_groups[0]['lr'], decimals=6))
        self.print_to_log_file("lr:", np.round(self.optimizer[1].param_groups[0]['lr'], decimals=6))

    def manage_patience(self):
        # update patience
        if self.patience is not None:
            # if best_MA_tr_loss_for_patience and best_epoch_based_on_MA_tr_loss were not yet initialized,
            # initialize them
            if self.best_MA_tr_loss_for_patience is None:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA

            if self.best_epoch_based_on_MA_tr_loss is None:
                self.best_epoch_based_on_MA_tr_loss = self.epoch

            if self.best_val_eval_criterion_MA is None:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA

            # check if the current epoch is the best one according to moving average of validation criterion. If so
            # then save 'best' model
            # Do not use this for validation. This is intended for test set prediction only.
            #self.print_to_log_file("current best_val_eval_criterion_MA is %.4f0" % self.best_val_eval_criterion_MA)
            #self.print_to_log_file("current val_eval_criterion_MA is %.4f" % self.val_eval_criterion_MA)

            if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
                #self.print_to_log_file("saving best epoch checkpoint...")
                if self.save_best_checkpoint: self.save_checkpoint(join(self.output_folder, "model_best.model"))

            # Now see if the moving average of the train loss has improved. If yes then reset patience, else
            # increase patience
            if self.train_loss_MA + self.train_loss_MA_eps < self.best_MA_tr_loss_for_patience:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA
                self.best_epoch_based_on_MA_tr_loss = self.epoch
                #self.print_to_log_file("New best epoch (train loss MA): %03.4f" % self.best_MA_tr_loss_for_patience)
            else:
                pass
                #self.print_to_log_file("No improvement: current train MA %03.4f, best: %03.4f, eps is %03.4f" %
                #                       (self.train_loss_MA, self.best_MA_tr_loss_for_patience, self.train_loss_MA_eps))

            # if patience has reached its maximum then finish training (provided lr is low enough)
            if self.epoch - self.best_epoch_based_on_MA_tr_loss > self.patience:
                # TODO: fix condition when optimizer is a type of tuple
                if isinstance(self.optimizer, tuple):
                    optimizer = self.optimizer[0]
                else:
                    optimizer = self.optimizer
                if optimizer.param_groups[0]['lr'] > self.lr_threshold:
                    #self.print_to_log_file("My patience ended, but I believe I need more time (lr > 1e-6)")
                    self.best_epoch_based_on_MA_tr_loss = self.epoch - self.patience // 2
                else:
                    #self.print_to_log_file("My patience ended")
                    continue_training = False
            else:
                pass
                #self.print_to_log_file(
                #    "Patience: %d/%d" % (self.epoch - self.best_epoch_based_on_MA_tr_loss, self.patience))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """

        # network_trainer's on_epoch_end
        self.finish_online_evaluation()  # does not have to do anything, but can be used to update self.all_val_eval_metrics

        self.plot_progress()

        self.maybe_update_lr()

        self.maybe_save_checkpoint()

        self.update_eval_criterion_MA()

        _ = self.manage_patience()  # Overwritten

        continue_training = self.epoch < self.max_num_epochs


        # from nnUNetTrainerV2's on_epoch_end
        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer[0].param_groups[0]["momentum"] = 0.95
                self.optimizer[1].param_groups[0]["momentum"] = 0.95
                self.network[0].apply(InitWeights_He(1e-2))
                self.network[1].apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def save_debug_information(self):
        # saving some debug information
        dct = OrderedDict()
        for k in self.__dir__():
            if not k.startswith("__"):
                if not callable(getattr(self, k)):
                    dct[k] = str(getattr(self, k))
        del dct['plans']
        del dct['intensity_properties']
        del dct['dataset']
        del dct['dataset_tr']
        del dct['dataset_val']
        save_json(dct, join(self.output_folder, "debug.json"))

        import shutil

        shutil.copyfile(self.plans_file, join(self.output_folder_base, "plans.pkl"))  # replace copy with copyfile

    def plot_network_architecture(self):
        try:
            from batchgenerators.utilities.file_and_folder_operations import join
            import hiddenlayer as hl
            if torch.cuda.is_available():
                g_1 = hl.build_graph(self.network[0], torch.rand((1, self.num_input_channels, *self.patch_size)).cuda(),
                                   transforms=None)
                g_2 = hl.build_graph(self.network[1], torch.rand((1, self.num_input_channels, *self.patch_size)).cuda(),
                                     transforms=None)
            else:
                g_1 = hl.build_graph(self.network[0], torch.rand((1, self.num_input_channels, *self.patch_size)),
                                     transforms=None)
                g_2 = hl.build_graph(self.network[1], torch.rand((1, self.num_input_channels, *self.patch_size)),
                                   transforms=None)
            g_1.save(join(self.output_folder, "network_architecture_1.pdf"))
            g_2.save(join(self.output_folder, "network_architecture_2.pdf"))
            del g_1, g_2
        except Exception as e:
            self.print_to_log_file("Unable to plot network architecture:")
            self.print_to_log_file(e)

            self.print_to_log_file("\nprinting the network instead:\n")
            self.print_to_log_file(self.network)
            self.print_to_log_file("\n")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def save_checkpoint(self, fname, save_optimizer=True):
        # Overwrite from NetworkTrainer
        start_time = time()
        state_dict_1 = self.network[0].state_dict()
        state_dict_2 = self.network[1].state_dict()
        for key in state_dict_1.keys():
            state_dict_1[key] = state_dict_1[key].cpu()
        for key in state_dict_2.keys():
            state_dict_2[key] = state_dict_2[key].cpu()

        lr_sched_state_dct_1 = None
        lr_sched_state_dct_2 = None
        # not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
        if self.lr_scheduler[0] is not None and hasattr(self.lr_scheduler[0], 'state_dict'):  # not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
            lr_sched_state_dct_1 = self.lr_scheduler[0].state_dict()
        if self.lr_scheduler[1] is not None and hasattr(self.lr_scheduler[1], 'state_dict'):  # not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
            lr_sched_state_dct_2 = self.lr_scheduler[1].state_dict()

        if save_optimizer:
            optimizer_state_dict_1 = self.optimizer[0].state_dict()
            optimizer_state_dict_2 = self.optimizer[1].state_dict()
        else:
            optimizer_state_dict_1 = None
            optimizer_state_dict_2 = None

        self.print_to_log_file("saving checkpoint...")
        save_this = {
            'epoch': self.epoch + 1,
            'state_dict': (state_dict_1, state_dict_2),
            'optimizer_state_dict': (optimizer_state_dict_1, optimizer_state_dict_2),
            'lr_scheduler_state_dict': (lr_sched_state_dct_1, lr_sched_state_dct_2),
            'plot_stuff': (self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode,
                           self.all_val_eval_metrics),
            'best_stuff' : (self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA)}
        if self.amp_grad_scaler is not None:
            save_this['amp_grad_scaler'] = self.amp_grad_scaler.state_dict()

        torch.save(save_this, fname)
        self.print_to_log_file("done, saving took %.2f seconds" % (time() - start_time))


        # append from nnUNetTrainerV2
        info = OrderedDict()
        info['init'] = self.init_args
        info['name'] = self.__class__.__name__
        info['class'] = str(self.__class__)
        info['plans'] = self.plans

        write_pickle(info, fname + ".pkl")

    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        Overwrite from NetworkTrainer, 有好几个load相关的方法，但都是基于这个方法的，重写它就好了
        used for if the checkpoint is already in ram
        :param checkpoint:
        :param train:
        :return:
        """
        if not self.was_initialized:
            self.initialize(train)

        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network[0].state_dict().keys())
        # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in checkpoint['state_dict'][0].items():
            key = k
            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.network[0].load_state_dict(new_state_dict)

        # load again
        print("If CPS is applied, checkpoint should be different!")
        curr_state_dict_keys = list(self.network[1].state_dict().keys())
        # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = OrderedDict()
        for k, value in checkpoint['state_dict'][1].items():
            key = k
            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.network[1].load_state_dict(new_state_dict)

        if self.fp16:
            self._maybe_init_amp()
            if 'amp_grad_scaler' in checkpoint.keys():
                self.amp_grad_scaler.load_state_dict(checkpoint['amp_grad_scaler'])

        self.epoch = checkpoint['epoch']
        if train:
            optimizer_state_dict_1 = checkpoint['optimizer_state_dict'][0]
            optimizer_state_dict_2 = checkpoint['optimizer_state_dict'][1]
            if optimizer_state_dict_1 is not None:
                self.optimizer[0].load_state_dict(optimizer_state_dict_1)
            if optimizer_state_dict_2 is not None:
                self.optimizer[1].load_state_dict(optimizer_state_dict_2)

            if self.lr_scheduler[0] is not None and hasattr(self.lr_scheduler[0], 'load_state_dict') and checkpoint[
                'lr_scheduler_state_dict'] is not None:
                self.lr_scheduler[0].load_state_dict(checkpoint['lr_scheduler_state_dict'][0])

            if self.lr_scheduler[1] is not None and hasattr(self.lr_scheduler[1], 'load_state_dict') and checkpoint[
                'lr_scheduler_state_dict'] is not None:
                self.lr_scheduler[1].load_state_dict(checkpoint['lr_scheduler_state_dict'][1])

            if issubclass(self.lr_scheduler[0].__class__, _LRScheduler):
                self.lr_scheduler[0].step(self.epoch)
            if issubclass(self.lr_scheduler[1].__class__, _LRScheduler):
                self.lr_scheduler[1].step(self.epoch)

        self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics = checkpoint[
            'plot_stuff']

        # load best loss (if present)
        if 'best_stuff' in checkpoint.keys():
            self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA = \
            checkpoint[
                'best_stuff']

        # after the training is done, the epoch is incremented one more time in my old code. This results in
        # self.epoch = 1001 for old trained models when the epoch is actually 1000. This causes issues because
        # len(self.all_tr_losses) = 1000 and the plot function will fail. We can easily detect and correct that here
        if self.epoch != len(self.all_tr_losses):
            self.print_to_log_file("WARNING in loading checkpoint: self.epoch != len(self.all_tr_losses). This is "
                                   "due to an old bug and should only appear when you are loading old models. New "
                                   "models should have this fixed! self.epoch is now set to len(self.all_tr_losses)")
            self.epoch = len(self.all_tr_losses)
            self.all_tr_losses = self.all_tr_losses[:self.epoch]
            self.all_val_losses = self.all_val_losses[:self.epoch]
            self.all_val_losses_tr_mode = self.all_val_losses_tr_mode[:self.epoch]
            self.all_val_eval_metrics = self.all_val_eval_metrics[:self.epoch]

        self._maybe_init_amp()

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network[0].do_ds
        self.network[0].do_ds = True
        self.network[1].do_ds = True

        ###########################################################
        # Overwrite super().run_training()
        # ret = super().run_training()

        self.save_debug_information()  # copy from nnUNetTrainer

        _ = self.tr_labeled_gen.next()
        _ = self.tr_unlabeled_gen.next()
        _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)
        self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []

            # set cps loss lambda
            if self.warm_up_lambda:
                if self.epoch < 500:
                    self.ssl_loss_weight = (self.epoch + 1) / 500 * 0.5
                else:
                    self.ssl_loss_weight = 0.5
            else:
                self.ssl_loss_weight = 0.5

            self.print_to_log_file("ssl_loss_weight : %f" % self.ssl_loss_weight)

            # train one epoch
            self.network[0].train()
            self.network[1].train()

            if self.use_progress_bar:
                with trange(self.num_batches_per_epoch) as tbar:
                    for b in tbar:
                        tbar.set_description("Epoch {}/{}".format(self.epoch+1, self.max_num_epochs))

                        l = self.run_iteration((self.tr_labeled_gen, self.tr_unlabeled_gen), True, train=True)

                        tbar.set_postfix(loss=l)
                        train_losses_epoch.append(l)
            else:
                for _ in range(self.num_batches_per_epoch):
                    l = self.run_iteration((self.tr_labeled_gen, self.tr_unlabeled_gen), True, train=True)
                    train_losses_epoch.append(l)

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])

            # validation
            with torch.no_grad():
                # validation with train=False
                self.network[0].eval()
                self.network[1].eval()
                val_losses = []
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.val_gen, False, True, train=False)
                    val_losses.append(l)
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])

                if self.also_val_in_tr_mode:
                    self.network[0].train()
                    self.network[1].train()
                    # validation with train=True
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        l = self.run_iteration(self.val_gen, False, train=False)
                        val_losses.append(l)
                    self.all_val_losses_tr_mode.append(np.mean(val_losses))
                    self.print_to_log_file("validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint: self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))

        ############################################################
        self.network[0].do_ds = ds
        self.network[1].do_ds = ds

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network[0].do_ds  # 两个network应该是相同的mode, 不用定义两个临时变量
        self.network[0].do_ds = False
        self.network[1].do_ds = False

        # Overwrite from nnUNetTrainer
        if pad_border_mode == 'constant' and pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        if do_mirroring and mirror_axes is None:
            mirror_axes = self.data_aug_params['mirror_axes']

        if do_mirroring:
            assert self.data_aug_params["do_mirror"], "Cannot do mirroring as test time augmentation when training " \
                                                      "was done without mirroring"

        valid = list((SegmentationNetwork, nn.DataParallel))
        assert isinstance(self.network[0], tuple(valid))
        assert isinstance(self.network[1], tuple(valid))

        current_mode = self.network[0].training

        self.network[0].eval()
        self.network[1].eval()

        # 只让network1推理即可
        ret = self.network[0].predict_3D(data, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                                      use_sliding_window=use_sliding_window, step_size=step_size,
                                      patch_size=self.patch_size, regions_class_order=self.regions_class_order,
                                      use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                                      pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                                      mixed_precision=mixed_precision)
        self.network[0].train(current_mode)
        self.network[1].train(current_mode)

        self.network[0].do_ds = ds
        self.network[1].do_ds = ds
        return ret

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        # copy from nnUNetTrainerV2
        ds = self.network[0].do_ds
        self.network[0].do_ds = False
        self.network[1].do_ds = False

        # copy from nnUNetTrainer
        current_mode = self.network[0].training
        self.network[0].eval()
        self.network[1].eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        # predictions as they come from the network go here
        output_folder = join(self.output_folder, validation_folder_name)
        maybe_mkdir_p(output_folder)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save_softmax': save_softmax,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []

        export_pool = Pool(default_num_threads)
        results = []

        for k in self.dataset_val.keys():
            properties = load_pickle(self.dataset[k]['properties_file'])
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            if overwrite or (not isfile(join(output_folder, fname + ".nii.gz"))) or \
                    (save_softmax and not isfile(join(output_folder, fname + ".npz"))):
                data = np.load(self.dataset[k]['data_file'])['data']

                print(k, data.shape)
                data[-1][data[-1] == -1] = 0

                softmax_pred = self.predict_preprocessed_data_return_seg_and_softmax(data[:-1],
                                                                                     do_mirroring=do_mirroring,
                                                                                     mirror_axes=mirror_axes,
                                                                                     use_sliding_window=use_sliding_window,
                                                                                     step_size=step_size,
                                                                                     use_gaussian=use_gaussian,
                                                                                     all_in_gpu=all_in_gpu,
                                                                                     mixed_precision=self.fp16)[1]

                softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                if save_softmax:
                    softmax_fname = join(output_folder, fname + ".npz")
                else:
                    softmax_fname = None

                """There is a problem with python process communication that prevents us from communicating objects
                larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
                communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long
                enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
                patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
                then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
                filename or np.ndarray and will handle this automatically"""
                if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                    np.save(join(output_folder, fname + ".npy"), softmax_pred)
                    softmax_pred = join(output_folder, fname + ".npy")

                results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                         ((softmax_pred, join(output_folder, fname + ".nii.gz"),
                                                           properties, interpolation_order, self.regions_class_order,
                                                           None, None,
                                                           softmax_fname, None, force_separate_z,
                                                           interpolation_order_z),
                                                          )
                                                         )
                               )

            pred_gt_tuples.append([join(output_folder, fname + ".nii.gz"),
                                   join(self.gt_niftis_folder, fname + ".nii.gz")])

        _ = [i.get() for i in results]
        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                             json_output_file=join(output_folder, "summary.json"),
                             json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads)

        if run_postprocessing_on_folds:
            # in the old nnunet we would stop here. Now we add a postprocessing. This postprocessing can remove everything
            # except the largest connected component for each class. To see if this improves results, we do this for all
            # classes and then rerun the evaluation. Those classes for which this resulted in an improved dice score will
            # have this applied during inference as well
            self.print_to_log_file("determining postprocessing")
            determine_postprocessing(self.output_folder, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug)
            # after this the final predictions for the vlaidation set can be found in validation_folder_name_base + "_postprocessed"
            # They are always in that folder, even if no postprocessing as applied!

        # detemining postprocesing on a per-fold basis may be OK for this fold but what if another fold finds another
        # postprocesing to be better? In this case we need to consolidate. At the time the consolidation is going to be
        # done we won't know what self.gt_niftis_folder was, so now we copy all the niftis into a separate folder to
        # be used later
        gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
        maybe_mkdir_p(gt_nifti_folder)
        for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
            success = False
            attempts = 0
            e = None
            while not success and attempts < 10:
                try:
                    shutil.copy(f, gt_nifti_folder)
                    success = True
                except OSError as e:
                    attempts += 1
                    sleep(1)
            if not success:
                print("Could not copy gt nifti file %s into folder %s" % (f, gt_nifti_folder))
                if e is not None:
                    raise e

        self.network[0].train(current_mode)
        self.network[1].train(current_mode)


        self.network[0].do_ds = ds
        self.network[1].do_ds = ds

    def preprocess_predict_nifti(self, input_files: List[str], output_file: str = None,
                                 softmax_ouput_file: str = None, mixed_precision: bool = True) -> None:
        # 好像是已经废弃的代码
        raise NotImplemented
