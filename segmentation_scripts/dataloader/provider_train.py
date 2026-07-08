from __future__ import absolute_import, division, print_function

import os
import random
import sys

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.aff_util import seg_to_affgraph
from utils.augmentation import ElasticAugment as Elastic
from utils.augmentation import Rescale
from utils.augmentation import SimpleAugment as Flip
from utils.consistency_aug_perturbations import (
    Artifact,
    BlurEnhanced,
    Cutout,
    GaussBlur,
    GaussNoise,
    Intensity,
    Missing,
    Mixup,
    SobelFilter,
)
from utils.seg_util import genSegMalis, mknhood3d


MODEL_CROP_SETTINGS = {
    "superhuman": [16, 160, 160],
    "SwinUNETR": [64, 96, 96],
    "SegMamba": [32, 128, 128],
}

DATASETS = {
    "AC3": ("AC3AC4", ["AC3_inputs.h5"], ["AC3_labels.h5"]),
    "AC4": ("AC3AC4", ["AC4_inputs.h5"], ["AC4_labels.h5"]),
    "t-raw": ("generate_AC", ["AC4_inputs_512.h5"], ["AC4_labels_512.h5"]),
    "t-NeuroDiff": (
        "generate_AC",
        ["AC4_inputs_512.h5", "mask-t_NeuroDiff.h5"],
        ["AC4_labels_512.h5", "AC4_labels-t.h5"],
    ),
}


def get_crop_size(model_type):
    if model_type not in MODEL_CROP_SETTINGS:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            f"Choose from {sorted(MODEL_CROP_SETTINGS)}."
        )
    return MODEL_CROP_SETTINGS[model_type]


def get_dataset_files(dataset_name):
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unsupported dataset_name '{dataset_name}'. "
            f"Choose from {sorted(DATASETS)}."
        )
    return DATASETS[dataset_name]


def load_h5(path):
    with h5py.File(path, "r") as f:
        return f["main"][:]


def crop_center_xy(volume, target_size):
    if target_size is None:
        return volume

    target_size = int(target_size)
    _, height, width = volume.shape
    if height == target_size and width == target_size:
        return volume
    if height < target_size or width < target_size:
        raise ValueError(
            f"Cannot crop volume with shape {volume.shape} to xy size {target_size}."
        )

    y0 = (height - target_size) // 2
    x0 = (width - target_size) // 2
    return volume[:, y0:y0 + target_size, x0:x0 + target_size]


def crop_padding_xy(volume, padding_xy):
    if padding_xy <= 0:
        return volume
    return volume[:, padding_xy:-padding_xy, padding_xy:-padding_xy]


def label_to_affinity(label, output_channels):
    if output_channels != 3:
        raise ValueError("This open-source training pipeline expects MODEL.output_nc = 3.")

    label = genSegMalis(label, 1)
    return seg_to_affgraph(label, mknhood3d(1), pad="replicate").astype(np.float32)


class Train(Dataset):
    def __init__(self, cfg):
        super(Train, self).__init__()
        self.cfg = cfg
        self.model_type = cfg.MODEL.model_type
        self.crop_size = get_crop_size(self.model_type)
        self.net_padding = [0, 0, 0]
        self.out_size = self.crop_size

        sub_path, self.train_datasets, self.train_labels = get_dataset_files(cfg.DATA.dataset_name)
        self.folder_name = os.path.join(cfg.DATA.data_folder, sub_path)
        self.train_split = cfg.DATA.train_split

        self.dataset = []
        self.labels = []
        for image_name, label_name in zip(self.train_datasets, self.train_labels):
            print(f"load {image_name} ...")
            data = load_h5(os.path.join(self.folder_name, image_name))
            label = load_h5(os.path.join(self.folder_name, label_name))

            data = self.apply_split(data)
            label = self.apply_split(label)
            data = crop_center_xy(data, cfg.DATA.label_crop_size)
            label = crop_center_xy(label, cfg.DATA.label_crop_size)

            self.dataset.append(data)
            self.labels.append(label)

        self.pad_short_z()
        self.raw_data_shape = [list(data.shape) for data in self.dataset]
        print("raw data shape:", self.raw_data_shape)

        self.sub_padding = [0, 80, 80]
        self.crop_from_origin = [
            self.crop_size[i] + 2 * self.sub_padding[i]
            for i in range(len(self.crop_size))
        ]

        self.per_mode = cfg.DATA.per_mode
        self.if_scale_aug = cfg.DATA.if_scale_aug
        self.if_intensity_aug = cfg.DATA.if_intensity_aug
        self.if_noise_aug = cfg.DATA.if_noise_aug
        self.if_blur_aug = cfg.DATA.if_blur_aug
        self.if_mask_aug = cfg.DATA.if_mask_aug
        self.if_sobel_aug = cfg.DATA.if_sobel_aug
        self.if_mixup_aug = cfg.DATA.if_mixup_aug
        self.if_misalign_aug = cfg.DATA.if_misalign_aug
        self.if_elastic_aug = cfg.DATA.if_elastic_aug
        self.if_artifact_aug = cfg.DATA.if_artifact_aug
        self.if_missing_aug = cfg.DATA.if_missing_aug
        self.if_blurenhanced_aug = cfg.DATA.if_blurenhanced_aug

        self.scale_factor = cfg.DATA.scale_factor
        self.min_noise_std = cfg.DATA.min_noise_std
        self.max_noise_std = cfg.DATA.max_noise_std
        self.min_kernel_size = cfg.DATA.min_kernel_size
        self.max_kernel_size = cfg.DATA.max_kernel_size
        self.min_sigma = cfg.DATA.min_sigma
        self.max_sigma = cfg.DATA.max_sigma

        self.simple_aug = Flip()
        self.perturbations_init()

    def apply_split(self, volume):
        if self.train_split is None:
            return volume
        if self.train_split > 0:
            return volume[: self.train_split]
        return volume[self.train_split :]

    def pad_short_z(self):
        for i, data in enumerate(self.dataset):
            num_z = data.shape[0]
            if num_z >= self.crop_size[0]:
                continue

            pad_left = (self.crop_size[0] - num_z) // 2
            pad_right = self.crop_size[0] - num_z - pad_left
            pad_width = ((pad_left, pad_right), (0, 0), (0, 0))
            self.dataset[i] = np.pad(self.dataset[i], pad_width, mode="reflect")
            self.labels[i] = np.pad(self.labels[i], pad_width, mode="reflect")

    def __getitem__(self, index):
        dataset_index = random.randint(0, len(self.train_datasets) - 1)
        image_volume = self.dataset[dataset_index]
        label_volume = self.labels[dataset_index]
        data_shape = self.raw_data_shape[dataset_index]

        random_z = random.randint(0, data_shape[0] - self.crop_from_origin[0])
        random_y = random.randint(0, data_shape[1] - self.crop_from_origin[1])
        random_x = random.randint(0, data_shape[2] - self.crop_from_origin[2])

        imgs = image_volume[
            random_z:random_z + self.crop_from_origin[0],
            random_y:random_y + self.crop_from_origin[1],
            random_x:random_x + self.crop_from_origin[2],
        ].copy()
        label = label_volume[
            random_z:random_z + self.crop_from_origin[0],
            random_y:random_y + self.crop_from_origin[1],
            random_x:random_x + self.crop_from_origin[2],
        ].copy()

        imgs = imgs.astype(np.float32) / 255.0
        imgs, label = self.simple_aug([imgs, label])
        imgs, label, _, _, _ = self.apply_perturbations(imgs, label, None, self.per_mode)

        label_affs = label_to_affinity(label, self.cfg.MODEL.output_nc)
        weightmap = self.build_weight_map(label_affs)

        imgs = imgs[np.newaxis, ...]
        return (
            np.ascontiguousarray(imgs, dtype=np.float32),
            np.ascontiguousarray(label_affs, dtype=np.float32),
            np.ascontiguousarray(weightmap, dtype=np.float32),
        )

    def build_weight_map(self, label_affs):
        weight_factor = np.sum(label_affs) / np.size(label_affs)
        weight_factor = np.clip(weight_factor, 1e-3, 1)
        return label_affs * (1 - weight_factor) / weight_factor + (1 - label_affs)

    def perturbations_init(self):
        self.per_rescale = Rescale(scale_factor=self.scale_factor, det_shape=self.crop_size)
        self.per_intensity = Intensity()
        self.per_gaussnoise = GaussNoise(
            min_std=self.min_noise_std,
            max_std=self.max_noise_std,
            norm_mode="trunc",
        )
        self.per_gaussblur = GaussBlur(
            min_kernel=self.min_kernel_size,
            max_kernel=self.max_kernel_size,
            min_sigma=self.min_sigma,
            max_sigma=self.max_sigma,
        )
        self.per_cutout = Cutout(model_type=self.model_type)
        self.per_sobel = SobelFilter(if_mean=True)
        self.per_mixup = Mixup(min_alpha=0.1, max_alpha=0.4)
        self.per_misalign = Elastic(
            control_point_spacing=[4, 40, 40],
            jitter_sigma=[0, 0, 0],
            prob_slip=0.2,
            prob_shift=0.2,
            max_misalign=17,
            padding=20,
        )
        self.per_elastic = Elastic(
            control_point_spacing=[4, 40, 40],
            jitter_sigma=[0, 2, 2],
            padding=20,
        )
        self.per_artifact = Artifact(min_sec=1, max_sec=5)
        self.per_missing = Missing(miss_fully_ratio=0.2, miss_part_ratio=0.5)
        self.per_blurenhanced = BlurEnhanced(blur_fully_ratio=0.5, blur_part_ratio=0.7)

    def apply_perturbations(self, data, mask, auxi=None, mode=1):
        if mode != 1:
            raise NotImplementedError("Only per_mode=1 is supported.")

        perturbations = [
            self.if_scale_aug,
            False,
            False,
            self.if_intensity_aug,
            self.if_noise_aug,
            self.if_blur_aug,
            self.if_mask_aug,
            self.if_sobel_aug,
            self.if_mixup_aug,
            self.if_misalign_aug,
            self.if_elastic_aug,
            self.if_artifact_aug,
            self.if_missing_aug,
            self.if_blurenhanced_aug,
        ]
        used = [idx for idx, enabled in enumerate(perturbations) if enabled]

        if not used:
            data = crop_padding_xy(data, self.sub_padding[-1])
            mask = crop_padding_xy(mask, self.sub_padding[-1])
            return data, mask, data.shape[-1], np.zeros(4, dtype=np.int32), 0

        rand_per = random.choice(used)
        if rand_per == 0:
            data, mask, scale_size = self.per_rescale(data, mask)
        else:
            data = crop_padding_xy(data, self.sub_padding[-1])
            mask = crop_padding_xy(mask, self.sub_padding[-1])
            scale_size = data.shape[-1]

        if rand_per == 3:
            data = self.per_intensity(data)
        elif rand_per == 4:
            data = self.per_gaussnoise(data)
        elif rand_per == 5:
            data = self.per_gaussblur(data)
        elif rand_per == 6:
            data = self.per_cutout(data)
        elif rand_per == 7:
            data = self.per_sobel(data)
        elif rand_per == 8 and auxi is not None:
            data = self.per_mixup(data, auxi)
        elif rand_per == 9:
            data, mask = self.per_misalign(data, mask)
        elif rand_per == 10:
            data, mask = self.per_elastic(data, mask)
        elif rand_per == 11:
            data = self.per_artifact(data)
        elif rand_per == 12:
            data = self.per_missing(data)
        elif rand_per == 13:
            data = self.per_blurenhanced(data)

        return data, mask, scale_size, np.zeros(4, dtype=np.int32), 0

    def __len__(self):
        return sys.maxsize


class Provider(object):
    def __init__(self, stage, cfg):
        if stage != "train":
            raise ValueError("Provider only supports stage='train'.")

        self.stage = stage
        self.data = Train(cfg)
        self.batch_size = cfg.TRAIN.batch_size
        self.num_workers = cfg.TRAIN.num_workers
        self.is_cuda = cfg.TRAIN.if_cuda
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1

    def __len__(self):
        return len(self.data)

    def build(self):
        self.data_iter = iter(
            DataLoader(
                dataset=self.data,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
            )
        )

    def next(self):
        if self.data_iter is None:
            self.build()

        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.epoch += 1
            self.build()
            batch = next(self.data_iter)

        self.iteration += 1
        if self.is_cuda:
            batch = [item.cuda() for item in batch]
        return batch
