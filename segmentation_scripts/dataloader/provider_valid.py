from __future__ import absolute_import, division, print_function

import os

import h5py
import numpy as np
from torch.utils.data import Dataset

from utils.aff_util import seg_to_affgraph
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


def label_to_affinity(label, output_channels):
    if output_channels != 3:
        raise ValueError("This open-source inference pipeline expects MODEL.output_nc = 3.")

    label = genSegMalis(label, 1)
    return seg_to_affgraph(label, mknhood3d(1), pad="replicate").astype(np.float32)


def crop_by_padding(volume, padding):
    z_pad, y_pad, x_pad = padding
    z_slice = slice(z_pad, -z_pad) if z_pad > 0 else slice(None)
    y_slice = slice(y_pad, -y_pad) if y_pad > 0 else slice(None)
    x_slice = slice(x_pad, -x_pad) if x_pad > 0 else slice(None)
    return volume[..., z_slice, y_slice, x_slice]


class Provider_valid(Dataset):
    def __init__(self, cfg, valid_data=None, num_z=18, test=False, test_split=None):
        self.cfg = cfg
        self.model_type = cfg.MODEL.model_type
        self.crop_size = get_crop_size(self.model_type)
        self.out_size = self.crop_size
        self.output_channel = cfg.MODEL.output_nc
        self.num_z = num_z
        self.test = test

        self.dataset_name = valid_data if valid_data is not None else cfg.DATA.dataset_name
        print("valid dataset:", self.dataset_name)

        sub_path, self.train_datasets, self.train_labels = get_dataset_files(self.dataset_name)
        self.folder_name = os.path.join(cfg.DATA.data_folder, sub_path)
        self.test_split = cfg.DATA.test_split if test_split is None else test_split
        print("the number of valid(test) = %d" % self.test_split)

        self.dataset = []
        self.labels = []
        for image_name, label_name in zip(self.train_datasets, self.train_labels):
            print(f"load {image_name} ...")
            data = load_h5(os.path.join(self.folder_name, image_name))
            label = load_h5(os.path.join(self.folder_name, label_name))
            self.dataset.append(self.apply_split(data))
            self.labels.append(self.apply_split(label))

        self.origin_data_shape = list(self.dataset[0].shape)
        self.gt_affs = [
            label_to_affinity(label.copy(), self.output_channel)
            for label in self.labels
        ]

        self.stride, self.valid_padding, self.num_zyx = self.get_inference_grid()
        self.pad_dataset()
        self.raw_data_shape = list(self.dataset[0].shape)
        self.reset_output()
        self.weight_vol = self.get_weight()
        self.num_per_dataset = self.num_zyx[0] * self.num_zyx[1] * self.num_zyx[2]
        self.iters_num = self.num_per_dataset * len(self.dataset)

    def apply_split(self, volume):
        if self.test_split is None:
            return volume
        if self.test_split > 0:
            return volume[: self.test_split]
        return volume[self.test_split :]

    def get_inference_grid(self):
        depth = self.dataset[0].shape[0]

        if self.model_type == "superhuman":
            stride = [10, 80, 80]
            padding_xy = 48
            num_xy = 13
            if depth == 50:
                return stride, [14, padding_xy, padding_xy], [7, num_xy, num_xy]
            if depth == 100:
                return stride, [13, padding_xy, padding_xy], [12, num_xy, num_xy]
            if depth == 104:
                return stride, [12, padding_xy, padding_xy], [12, num_xy, num_xy]
            if depth == 25:
                return [15, 80, 80], [4, padding_xy, padding_xy], [2, num_xy, num_xy]

        if self.model_type == "SwinUNETR":
            stride = [32, 48, 48]
            padding_xy = 40
            num_xy = 22
            if depth == 50:
                return stride, [7, padding_xy, padding_xy], [1, num_xy, num_xy]
            if depth == 100:
                return stride, [14, padding_xy, padding_xy], [3, num_xy, num_xy]
            if depth == 200:
                return stride, [12, padding_xy, padding_xy], [6, num_xy, num_xy]
            if depth == 520:
                return stride, [12, padding_xy, padding_xy], [16, num_xy, num_xy]

        if self.model_type == "SegMamba":
            stride = [16, 64, 64]
            padding_xy = 32
            num_xy = 16
            if depth == 50:
                return stride, [7, padding_xy, padding_xy], [3, num_xy, num_xy]
            if depth == 100:
                return stride, [14, padding_xy, padding_xy], [7, num_xy, num_xy]
            if depth == 200:
                return stride, [12, padding_xy, padding_xy], [13, num_xy, num_xy]
            if depth == 520:
                return stride, [12, padding_xy, padding_xy], [33, num_xy, num_xy]

        raise NotImplementedError(
            f"No inference grid for model_type={self.model_type}, depth={depth}."
        )

    def pad_dataset(self):
        pad_width = (
            (self.valid_padding[0], self.valid_padding[0]),
            (self.valid_padding[1], self.valid_padding[1]),
            (self.valid_padding[2], self.valid_padding[2]),
        )
        for i in range(len(self.dataset)):
            self.dataset[i] = np.pad(self.dataset[i], pad_width, mode="reflect")
            self.labels[i] = np.pad(self.labels[i], pad_width, mode="reflect")

    def __getitem__(self, index):
        pos_data = index // self.num_per_dataset
        local_index = index % self.num_per_dataset
        pos_z = local_index // (self.num_zyx[1] * self.num_zyx[2])
        pos_xy = local_index % (self.num_zyx[1] * self.num_zyx[2])
        pos_y = pos_xy // self.num_zyx[2]
        pos_x = pos_xy % self.num_zyx[2]

        from_z = pos_z * self.stride[0]
        from_y = pos_y * self.stride[1]
        from_x = pos_x * self.stride[2]
        from_z, end_z = self.bound_patch(from_z, self.crop_size[0], self.raw_data_shape[0])
        from_y, end_y = self.bound_patch(from_y, self.crop_size[1], self.raw_data_shape[1])
        from_x, end_x = self.bound_patch(from_x, self.crop_size[2], self.raw_data_shape[2])
        self.pos = [from_z, from_y, from_x]

        imgs = self.dataset[pos_data][from_z:end_z, from_y:end_y, from_x:end_x].copy()
        label = self.labels[pos_data][from_z:end_z, from_y:end_y, from_x:end_x].copy()
        label_affs = label_to_affinity(label, self.output_channel)
        weightmap = self.build_weight_map(label_affs)

        imgs = imgs.astype(np.float32) / 255.0
        imgs = imgs[np.newaxis, ...]
        return (
            np.ascontiguousarray(imgs, dtype=np.float32),
            np.ascontiguousarray(label_affs, dtype=np.float32),
            np.ascontiguousarray(weightmap, dtype=np.float32),
        )

    @staticmethod
    def bound_patch(start, patch_size, full_size):
        end = start + patch_size
        if end > full_size:
            end = full_size
            start = end - patch_size
        return start, end

    def build_weight_map(self, label_affs):
        weight_factor = np.sum(label_affs) / np.size(label_affs)
        weight_factor = np.clip(weight_factor, 1e-3, 1)
        return label_affs * (1 - weight_factor) / weight_factor + (1 - label_affs)

    def __len__(self):
        return self.iters_num

    def reset_output(self):
        output_channels = 3 if self.test else self.output_channel
        self.out_affs = np.zeros([output_channels] + self.raw_data_shape, dtype=np.float32)
        self.weight_map = np.zeros([1] + self.raw_data_shape, dtype=np.float32)

    def get_weight(self, sigma=0.2, mu=0.0):
        zz, yy, xx = np.meshgrid(
            np.linspace(-1, 1, self.out_size[0], dtype=np.float32),
            np.linspace(-1, 1, self.out_size[1], dtype=np.float32),
            np.linspace(-1, 1, self.out_size[2], dtype=np.float32),
            indexing="ij",
        )
        dd = np.sqrt(zz * zz + yy * yy + xx * xx)
        return (1e-6 + np.exp(-((dd - mu) ** 2 / (2.0 * sigma ** 2))))[np.newaxis, ...]

    def add_vol(self, affs_vol):
        from_z, from_y, from_x = self.pos
        self.out_affs[
            :,
            from_z:from_z + self.out_size[0],
            from_y:from_y + self.out_size[1],
            from_x:from_x + self.out_size[2],
        ] += affs_vol * self.weight_vol
        self.weight_map[
            :,
            from_z:from_z + self.out_size[0],
            from_y:from_y + self.out_size[1],
            from_x:from_x + self.out_size[2],
        ] += self.weight_vol

    def get_results(self):
        self.out_affs = self.out_affs / self.weight_map
        self.out_affs = crop_by_padding(self.out_affs, self.valid_padding)
        return self.out_affs

    def get_gt_affs(self, num_data=0):
        return self.gt_affs[num_data].copy()

    def get_gt_lb(self, num_data=0):
        return crop_by_padding(self.labels[num_data].copy(), self.valid_padding)

    def get_raw_data(self, num_data=0):
        return crop_by_padding(self.dataset[num_data].copy(), self.valid_padding)
