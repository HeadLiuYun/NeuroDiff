# -*- coding:utf-8 -*-
import argparse
import numpy as np
from scipy.ndimage import map_coordinates, zoom
import os
from pathlib import Path
from PIL import Image
import h5py
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from skimage.draw import ellipse, polygon
from skimage.morphology import closing, disk, erosion, opening
from skimage.measure import regionprops, find_contours
from shapely.geometry import LineString
from scipy.ndimage import label, center_of_mass as scipy_center_of_mass
import random


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return str(path)
    return str(REPO_ROOT / path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate biology-guided NeuroDiff condition masks."
    )
    parser.add_argument(
        "--raw_label_path",
        type=str,
        default="./segmentation_scripts/data/AC3AC4/AC4_labels.h5",
        help="Input neuron instance label H5 file.",
    )
    parser.add_argument(
        "--mito_mask_dir",
        type=str,
        default="./data/EM/AC4-t/mito/",
        help="Folder containing mitochondria mask PNGs used as templates.",
    )
    parser.add_argument(
        "--elastic_label_path",
        type=str,
        default="./segmentation_scripts/data/generate_AC/AC4_labels-t.h5",
        help="Output H5 path for the elastically deformed labels.",
    )
    parser.add_argument(
        "--elastic_membrane_dir",
        type=str,
        default="./data/EM/AC4-t/affs-ela/",
        help="Output folder for elastically deformed membrane PNGs.",
    )
    parser.add_argument(
        "--output_condition_dir",
        type=str,
        default="./data/EM/AC4-t/mask-ela/",
        help="Output folder for remodeled condition mask PNGs.",
    )
    parser.add_argument("--crop_z_start", type=int, default=0)
    parser.add_argument("--crop_z_size", type=int, default=16)
    parser.add_argument("--crop_y_start", type=int, default=256)
    parser.add_argument("--crop_x_start", type=int, default=256)
    parser.add_argument("--crop_xy_size", type=int, default=512)
    parser.add_argument("--point_spacing", type=int, nargs=3, default=(4, 20, 20))
    parser.add_argument("--jitter_sigma", type=int, nargs=3, default=(0, 4, 4))
    parser.add_argument("--padding", type=int, default=20)
    parser.add_argument("--membrane_iteration", type=int, default=0)
    parser.add_argument("--select_neurons_num", type=int, default=20)
    parser.add_argument("--sample_neurons_num", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def mknhood3d(radius=1):
    ceilrad = np.ceil(radius)
    x = np.arange(-ceilrad, ceilrad + 1, 1)
    y = np.arange(-ceilrad, ceilrad + 1, 1)
    z = np.arange(-ceilrad, ceilrad + 1, 1)
    [i, j, k] = np.meshgrid(z, y, x)

    idxkeep = (i ** 2 + j ** 2 + k ** 2) <= radius ** 2
    i = i[idxkeep].ravel();
    j = j[idxkeep].ravel();
    k = k[idxkeep].ravel();
    zeroIdx = np.array(len(i) // 2).astype(np.int32);

    nhood = np.vstack((k[:zeroIdx], i[:zeroIdx], j[:zeroIdx])).T.astype(np.int32)
    return np.ascontiguousarray(np.flipud(nhood))


def genSegMalis(gg3, iter_num):  # given input seg map, widen the seg border
    gg3_dz = np.zeros(gg3.shape).astype(np.uint32)
    gg3_dz[1:, :, :] = (np.diff(gg3, axis=0))
    gg3_dy = np.zeros(gg3.shape).astype(np.uint32)
    gg3_dy[:, 1:, :] = (np.diff(gg3, axis=1))
    gg3_dx = np.zeros(gg3.shape).astype(np.uint32)
    gg3_dx[:, :, 1:] = (np.diff(gg3, axis=2))
    gg3g = ((gg3_dx + gg3_dy) > 0)
    # stel=np.array([[1, 1],[1,1]]).astype(bool)
    stel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(bool)
    # stel=np.array([[1,1,1,1],[1, 1, 1, 1],[1,1,1,1],[1,1,1,1]]).astype(bool)
    gg3gd = np.zeros(gg3g.shape)
    for i in range(gg3g.shape[0]):
        gg3gd[i, :, :] = binary_dilation(gg3g[i, :, :], structure=stel, iterations=iter_num)
    out = gg3.copy()
    out[gg3gd == 1] = 0
    return out


def seg_to_affgraph(seg, nhood, pad=''):
    shape = seg.shape
    nEdge = nhood.shape[0]
    aff = np.zeros((nEdge,) + shape, dtype=np.int32)

    for e in range(nEdge):
        aff[e, \
        max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]), \
        max(0, -nhood[e, 1]):min(shape[1], shape[1] - nhood[e, 1]), \
        max(0, -nhood[e, 2]):min(shape[2], shape[2] - nhood[e, 2])] = \
            (seg[max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]), \
             max(0, -nhood[e, 1]):min(shape[1], shape[1] - nhood[e, 1]), \
             max(0, -nhood[e, 2]):min(shape[2], shape[2] - nhood[e, 2])] == \
             seg[max(0, nhood[e, 0]):min(shape[0], shape[0] + nhood[e, 0]), \
             max(0, nhood[e, 1]):min(shape[1], shape[1] + nhood[e, 1]), \
             max(0, nhood[e, 2]):min(shape[2], shape[2] + nhood[e, 2])]) \
            * (seg[max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]), \
               max(0, -nhood[e, 1]):min(shape[1], shape[1] - nhood[e, 1]), \
               max(0, -nhood[e, 2]):min(shape[2], shape[2] - nhood[e, 2])] > 0) \
            * (seg[max(0, nhood[e, 0]):min(shape[0], shape[0] + nhood[e, 0]), \
               max(0, nhood[e, 1]):min(shape[1], shape[1] + nhood[e, 1]), \
               max(0, nhood[e, 2]):min(shape[2], shape[2] + nhood[e, 2])] > 0)
    if nEdge == 3 and pad == 'replicate':  # pad the boundary affinity
        aff[0, 0] = (seg[0] > 0).astype(aff.dtype)
        aff[1, :, 0] = (seg[:, 0] > 0).astype(aff.dtype)
        aff[2, :, :, 0] = (seg[:, :, 0] > 0).astype(aff.dtype)

    return aff


def upscale_transformation(transformation,
                           output_shape,
                           interpolate_order=1):
    input_shape = transformation.shape[1:]
    dims = len(output_shape)
    scale = tuple(float(s) / c for s, c in zip(output_shape, input_shape))

    scaled = np.zeros((dims,) + output_shape, dtype=np.float32)
    for d in range(dims):
        zoom(transformation[d], zoom=scale,
             output=scaled[d], order=interpolate_order)
    return scaled


def create_identity_transformation(shape, subsample=1):
    dims = len(shape)
    subsample_shape = tuple(max(1, int(s / subsample)) for s in shape)
    step_width = tuple(float(shape[d] - 1) / (subsample_shape[d] - 1)
                       if subsample_shape[d] > 1 else 1 for d in range(dims))

    axis_ranges = (
        np.arange(subsample_shape[d], dtype=np.float32) * step_width[d]
        for d in range(dims)
    )
    return np.array(np.meshgrid(*axis_ranges, indexing='ij'), dtype=np.float32)


def create_elastic_transformation(shape,
                                  control_point_spacing=100,
                                  jitter_sigma=10.0,
                                  subsample=1):
    dims = len(shape)
    subsample_shape = tuple(max(1, int(s / subsample)) for s in shape)

    try:
        spacing = tuple((d for d in control_point_spacing))
    except:
        spacing = (control_point_spacing,) * dims
    try:
        sigmas = [s for s in jitter_sigma]
    except:
        sigmas = [jitter_sigma] * dims

    control_points = tuple(
        max(1, int(round(float(shape[d]) / spacing[d])))
        for d in range(len(shape))
    )
    # jitter control points
    control_point_offsets = np.zeros(
        (dims,) + control_points, dtype=np.float32)
    for d in range(dims):
        if sigmas[d] > 0:
            control_point_offsets[d] = np.random.normal(scale=sigmas[d], size=control_points)
    transform = upscale_transformation(control_point_offsets, subsample_shape, interpolate_order=3)
    return transform


def apply_transformation(image,
                         transformation,
                         interpolate=True,
                         outside_value=0,
                         output=None):
    order = 1 if interpolate == True else 0
    output = image.dtype if output is None else output
    return map_coordinates(image,
                           transformation,
                           output=output,
                           order=order,
                           mode='constant',
                           cval=outside_value)


def elastic(img, point_spacing, jitter_sigma, padding):
    img = np.pad(img, ((0, 0), (padding, padding), (padding, padding)), mode='reflect')
    img_shape = img.shape
    transformation = create_identity_transformation(img_shape)
    transformation += create_elastic_transformation(img_shape, point_spacing, jitter_sigma)
    img_transform = apply_transformation(img, transformation, interpolate=False, outside_value=0,
                                         output=np.zeros(img.shape, dtype=np.float32))
    img_transform = img_transform[:, padding:-padding, padding:-padding]
    return img_transform


def count_neurons(data):
    unique_neurons = np.unique(data)

    if 0 in unique_neurons:
        unique_neurons = unique_neurons[unique_neurons != 0]

    neuron_sizes = {}
    for neuron in unique_neurons:
        voxel_count = np.sum(data == neuron)
        neuron_sizes[neuron] = voxel_count

    return neuron_sizes


# def calculate_center_of_mass_n(layer, max_neuron_id):
#     # 在某一层图像中，找到指定神经元 ID 所对应的连通区域中最大的一个，并返回它的质心（center of mass）坐标
#     labeled_layer, num_features = label(layer == max_neuron_id)
#
#     if num_features == 0:
#         return None
#
#     region_sizes = [np.sum(labeled_layer == i) for i in range(1, num_features + 1)]
#     largest_region_label = np.argmax(region_sizes) + 1
#
#     largest_region_size = region_sizes[largest_region_label - 1]
#     if largest_region_size < 5000:
#         return None
#
#     largest_region_mask = (labeled_layer == largest_region_label)
#
#     center_of_mass_coords = scipy_center_of_mass(largest_region_mask)
#     center_of_mass_coords = np.round(center_of_mass_coords).astype(int)
#
#     return tuple(center_of_mass_coords)

def calculate_center_of_mass_n(layer, max_neuron_id):
    labeled_layer, num_features = label(layer == max_neuron_id)

    if num_features == 0:
        return None, None

    region_sizes = [np.sum(labeled_layer == i) for i in range(1, num_features + 1)]
    largest_region_label = np.argmax(region_sizes) + 1

    largest_region_size = region_sizes[largest_region_label - 1]
    if largest_region_size < 5000:
        return None, None

    largest_region_mask = (labeled_layer == largest_region_label)

    center_of_mass_coords = scipy_center_of_mass(largest_region_mask)
    center_of_mass_coords = np.round(center_of_mass_coords).astype(int)

    return tuple(center_of_mass_coords), largest_region_mask


def rotate_coords(coords, angle, center):
    angle = -np.deg2rad(angle)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])

    coords = coords - center
    rotated_coords = np.dot(coords, rotation_matrix.T)
    rotated_coords = rotated_coords + center

    rotated_coords = np.round(rotated_coords).astype(int)

    return rotated_coords


def generate_rotated_mito_mask(center, axes, shape, angle):
    rr, cc = ellipse(center[0], center[1], axes[0], axes[1], shape=shape)

    coords = np.column_stack((rr, cc))

    rotated_coords = rotate_coords(coords, angle, center)

    rr_rotated, cc_rotated = rotated_coords[:, 0], rotated_coords[:, 1]

    rr_rotated = np.clip(rr_rotated, 0, shape[0] - 1)
    cc_rotated = np.clip(cc_rotated, 0, shape[1] - 1)

    mito_mask = np.zeros(shape, dtype=np.uint8)
    mito_mask[rr_rotated, cc_rotated] = 255

    mito_mask = closing(mito_mask, disk(1))

    return mito_mask


def generate_mito_mask(center, axes, shape):
    rr, cc = ellipse(center[0], center[1], axes[0], axes[1], shape=shape)

    rr = np.round(rr).astype(int)
    cc = np.round(cc).astype(int)

    mito_mask = np.zeros(shape, dtype=np.uint8)
    mito_mask[rr, cc] = 255

    return mito_mask


def read_png(path):
    files = sorted([f for f in os.listdir(path) if f.endswith('.png')])

    img_list = []
    for f in files:
        img = np.array(Image.open(os.path.join(path, f)).convert('L'))
        img_list.append(img)
    return np.stack(img_list)


def save_png(data, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for z in range(data.shape[0]):
        output_path = os.path.join(save_dir, f"{z:04d}.png")
        img = Image.fromarray(data[z, :, :])
        img = img.convert("L")
        img.save(output_path)


def read_h5(path):
    f_raw = h5py.File(os.path.join(path), 'r')
    data = f_raw['main'][:]
    f_raw.close()
    return data


def write_h5(path, data):
    f = h5py.File(os.path.join(path), 'w')
    f.create_dataset('main', data=data, dtype=data.dtype, compression='gzip')
    f.close()


def lb_to_aff(lb, output_path, iteration=0):
    os.makedirs(output_path, exist_ok=True)
    lb = genSegMalis(lb, 1)
    affs = seg_to_affgraph(lb, mknhood3d(1), pad='replicate').astype(np.float32)
    affs = (affs * 255).astype(np.uint8)

    print(affs.shape)
    output = np.transpose(affs, (1, 2, 3, 0))
    for j in range(affs.shape[1]):
        image_array = output[j]
        image_grey = np.minimum(image_array[:, :, 1], image_array[:, :, 2])
        image_grey = 255 - image_grey
        if iteration > 0:
            # 腐蚀一下膜 让膜变薄
            mask = (image_grey == 255)
            eroded_mask = binary_erosion(mask, structure=np.ones((3, 3)), iterations=iteration)
            image_eroded = eroded_mask.astype(np.uint8) * 255
            img = Image.fromarray(image_eroded)
        else:
            img = Image.fromarray(image_grey)
        img.save(os.path.join(output_path, f"{str(j).zfill(4)}.png"))


def has_png_stack(path, expected_count=None):
    if not os.path.isdir(path):
        return False
    png_files = [f for f in os.listdir(path) if f.endswith('.png')]
    if expected_count is not None:
        return len(png_files) >= expected_count
    return len(png_files) > 0


def ensure_elastic_label_and_membrane(
        raw_label_path,
        elastic_label_path,
        elastic_membrane_dir,
        crop_slices=None,
        point_spacing=(4, 20, 20),
        jitter_sigma=(0, 4, 4),
        padding=20,
        membrane_iteration=0):
    label_exists = os.path.exists(elastic_label_path)
    membrane_exists = has_png_stack(elastic_membrane_dir)

    if label_exists and membrane_exists:
        print(f"Skip elastic label and membrane: {elastic_label_path}, {elastic_membrane_dir}")
        return read_h5(elastic_label_path)

    if label_exists:
        print(f"Load existing elastic label: {elastic_label_path}")
        lb = read_h5(elastic_label_path)
    else:
        print(f"Generate elastic label: {elastic_label_path}")
        data = read_h5(raw_label_path)
        if crop_slices is not None:
            data = data[crop_slices]
        lb = elastic(data, point_spacing=point_spacing, jitter_sigma=jitter_sigma, padding=padding)
        lb = lb.astype(np.uint16)
        write_h5(elastic_label_path, lb)

    if membrane_exists:
        print(f"Skip elastic membrane: {elastic_membrane_dir}")
    else:
        print(f"Generate elastic membrane: {elastic_membrane_dir}")
        lb_to_aff(lb, elastic_membrane_dir, iteration=membrane_iteration)

    return lb


def lb_to_mask(data, select_neurons_num, sample_neurons_num, output_dir, membrane_dir, feature_ls, feature_angles,
               feature_sizes):
    neuron_sizes = count_neurons(data)
    sorted_neurons = sorted(neuron_sizes.items(), key=lambda x: x[1], reverse=True)[:select_neurons_num]

    selected_neurons = random.sample([neuron[0] for neuron in sorted_neurons], sample_neurons_num)

    mito_params = {}
    # for neuron_id in selected_neurons:
    #     # base_axes = (random.randint(15, 20), random.randint(15, 20))
    #     base_point = random.choice(feature_ls)
    #     base_angle = random.randint(0, 360)
    #     mito_params[neuron_id] = {"base_point": base_point, "base_angle": base_angle}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for z in range(data.shape[0]):

        layer_data = data[z, :, :]

        layer_masks = np.zeros_like(layer_data, dtype=np.uint8)

        for max_neuron_id in selected_neurons:
            # center_of_mass = calculate_center_of_mass_n(layer_data, max_neuron_id)
            center_of_mass, largest_region_mask = calculate_center_of_mass_n(layer_data, max_neuron_id)

            if center_of_mass is not None:
                if max_neuron_id not in mito_params:
                    # base_point = random.choice(feature_ls)
                    # base_angle = random.randint(0, 360)
                    region = regionprops(largest_region_mask.astype(int))[0]
                    region_angle = -np.rad2deg(region.orientation)
                    region_size = region.major_axis_length
                    base_point, base_angle = match_feature(region_angle, region_size, feature_ls, feature_angles,
                                                           feature_sizes)
                    if base_point is None:
                        continue
                    # base_angle = 90 + base_angle
                    mito_params[max_neuron_id] = {
                        "base_point": base_point,
                        "base_angle": base_angle
                    }
                base_point = mito_params[max_neuron_id]["base_point"]
                base_angle = mito_params[max_neuron_id]["base_angle"]

                # noise_range = 2  # 控制偏移范围，可调
                # point = base_point + np.random.randint(-noise_range, noise_range + 1, size=base_point.shape)
                # point[:, 0] = np.clip(point[:, 0], 0, layer_data.shape[0] - 1)
                # point[:, 1] = np.clip(point[:, 1], 0, layer_data.shape[1] - 1)
                point = base_point
                angle = base_angle + random.randint(-5, 5)

                mito_mask = reconstruct_mask_from_contour(point, layer_data.shape, center_of_mass, angle)

                mito_mask = opening(mito_mask, disk(3))
                mito_mask = closing(mito_mask, disk(3))

                # # === 防止越界部分 ===
                # neuron_mask = (layer_data == max_neuron_id).astype(np.uint8) * 255
                # neuron_mask_eroded = erosion(neuron_mask, disk(5))  # 调节 disk 半径
                # # 将线粒体 mask 限制在腐蚀后的 neuron 区域内
                # mito_mask = mito_mask * (neuron_mask_eroded > 0)

                # # === 防止越界部分 ===
                # neuron_mask = (layer_data == max_neuron_id).astype(np.uint8) * 255
                # neuron_mask_bin = neuron_mask > 0
                # mito_mask_bin = mito_mask > 0
                # max_iter = 20
                # i = 0
                # while not np.all(neuron_mask_bin[mito_mask_bin]):
                #     mito_mask_bin = erosion(mito_mask_bin, disk(1))
                #     i += 1
                #     # 检查是否断裂成多个连通区域
                #     labeled, num = label(mito_mask_bin)
                #     if num > 1:
                #         mito_mask_bin[:] = 0
                #         break
                #     if i >= max_iter or np.sum(mito_mask_bin) == 0:
                #         mito_mask_bin[:] = 0
                #         break
                # mito_mask = (mito_mask_bin.astype(np.uint8)) * 255

                neuron_mask = (layer_data == max_neuron_id).astype(np.uint8) * 255
                neuron_mask_bin = neuron_mask > 0
                mito_mask_bin = mito_mask > 0
                # mito中不在neuron中的像素数量
                outside_mask = mito_mask_bin & (~neuron_mask_bin)
                outside_ratio = np.sum(outside_mask) / np.sum(mito_mask_bin)
                # 设置容忍上限（比如5%）
                threshold = 0.2
                if outside_ratio > threshold:
                    mito_mask_bin[:] = 0
                mito_mask = mito_mask_bin.astype(np.uint8) * 255

                layer_masks = np.maximum(layer_masks, mito_mask)

        membrane_img = np.array(Image.open(os.path.join(membrane_dir, f"{z:04d}.png")))
        combined_img = np.zeros_like(membrane_img)
        combined_img[membrane_img == 255] = 1
        combined_img[layer_masks == 255] = 2

        output_path = os.path.join(output_dir, f"{z:04d}.png")
        img = Image.fromarray(combined_img)
        img = img.convert("L")
        img.save(output_path)


def simplify_contour(contour, tolerance=2.0):
    # 使用 RDP 算法简化轮廓
    line = LineString(contour)
    simplified = line.simplify(tolerance, preserve_topology=False)
    return np.array(simplified.coords)


def extract_mask_library(mask, min_area=100, tolerance=2.0):
    # 从2D mask中提取多个连通区域的简化轮廓 + 主轴角度 + 主轴长度
    mask = mask > 0
    mask = mask.astype(bool)
    labeled, _ = label(mask)

    feature_library = []
    feature_angles = []
    feature_sizes = []

    H, W = mask.shape

    for region in regionprops(labeled):
        if region.area < min_area:
            continue

        submask = labeled == region.label
        contours = find_contours(submask, level=0.5)
        if not contours:
            continue

        # 使用最大轮廓，忽略小孔
        contour = max(contours, key=len)

        # 跳过贴边轮廓
        if np.any(
                (contour[:, 0] <= 1) | (contour[:, 0] >= H - 2) |
                (contour[:, 1] <= 1) | (contour[:, 1] >= W - 2)
        ):
            continue

        simplified = simplify_contour(contour, tolerance=tolerance)

        # region.orientation 是弧度，逆时针为正；转换为度，记得取负号与图像方向统一
        angle = -np.rad2deg(region.orientation)
        size = region.major_axis_length

        feature_library.append(simplified)
        feature_angles.append(angle)
        feature_sizes.append(size)

    return feature_library, feature_angles, feature_sizes


def extract_mask_library_3d(mask_3d, min_area=100, tolerance=2.0):
    feature_library = []
    feature_angles = []
    feature_sizes = []

    for i in range(mask_3d.shape[0]):
        slice_img = mask_3d[i]
        contours, angles, sizes = extract_mask_library(slice_img, min_area=min_area, tolerance=tolerance)

        feature_library.extend(contours)
        feature_angles.extend(angles)
        feature_sizes.extend(sizes)

    return feature_library, feature_angles, feature_sizes


# def reconstruct_mask_from_contour(contour_points, image_shape, target_centroid):
#     # 计算原始质心（轮廓点的平均值）
#     orig_centroid = contour_points.mean(axis=0)  # (cy, cx)，浮点数
#
#     # 计算平移偏移
#     dy = target_centroid[0] - orig_centroid[0]
#     dx = target_centroid[1] - orig_centroid[1]
#
#     # 平移轮廓点，并四舍五入取整，转换为整数坐标
#     shifted_contour = contour_points + np.array([dy, dx])
#     shifted_contour = np.round(shifted_contour).astype(int)
#
#     # 限制轮廓点不要越界
#     shifted_contour[:, 0] = np.clip(shifted_contour[:, 0], 0, image_shape[0] - 1)
#     shifted_contour[:, 1] = np.clip(shifted_contour[:, 1], 0, image_shape[1] - 1)
#
#     # 拆解坐标
#     y_coords = shifted_contour[:, 0]
#     x_coords = shifted_contour[:, 1]
#
#     # 多边形填充
#     rr, cc = polygon(y_coords, x_coords, shape=image_shape)
#
#     # 初始化黑图，并填充区域
#     mask = np.zeros(image_shape, dtype=np.uint8)
#     mask[rr, cc] = 255
#
#     return mask

def reconstruct_mask_from_contour(contour_points, image_shape, target_centroid, angle=0):
    # 原始质心
    orig_centroid = contour_points.mean(axis=0)

    # 计算平移向量并执行平移
    shift = np.array(target_centroid) - orig_centroid
    shifted_contour = contour_points + shift

    # 旋转（使用你提供的 rotate_coords）
    rotated_contour = rotate_coords(shifted_contour, angle, center=np.array(target_centroid))

    # 裁剪边界
    rotated_contour[:, 0] = np.clip(rotated_contour[:, 0], 0, image_shape[0] - 1)
    rotated_contour[:, 1] = np.clip(rotated_contour[:, 1], 0, image_shape[1] - 1)

    # 填充 mask
    y_coords = rotated_contour[:, 0]
    x_coords = rotated_contour[:, 1]
    rr, cc = polygon(y_coords, x_coords, shape=image_shape)

    mask = np.zeros(image_shape, dtype=np.uint8)
    mask[rr, cc] = 255

    return mask


def match_feature(region_angle, region_size, feature_ls, feature_angles, feature_sizes, target_size_ratio=0.35,
                  size_margin=0.05):
    lower_bound = target_size_ratio - size_margin
    upper_bound = target_size_ratio + size_margin
    max_attempts = len(feature_ls)

    attempts = 0
    while attempts < max_attempts:
        idx = random.randint(0, len(feature_ls) - 1)
        feat = feature_ls[idx]
        angle = feature_angles[idx]
        size = feature_sizes[idx]

        size_ratio = size / region_size
        if lower_bound <= size_ratio <= upper_bound:
            # 计算角度偏差，方向保留正负值
            angle_offset = (region_angle - angle + 180) % 360 - 180
            return feat, angle_offset

        attempts += 1

    # 如果所有尝试都失败，返回 None
    return None, None


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.raw_label_path = resolve_path(args.raw_label_path)
    args.mito_mask_dir = resolve_path(args.mito_mask_dir)
    args.elastic_label_path = resolve_path(args.elastic_label_path)
    args.elastic_membrane_dir = resolve_path(args.elastic_membrane_dir)
    args.output_condition_dir = resolve_path(args.output_condition_dir)

    label_crop = np.s_[
        args.crop_z_start:args.crop_z_start + args.crop_z_size,
        args.crop_y_start:args.crop_y_start + args.crop_xy_size,
        args.crop_x_start:args.crop_x_start + args.crop_xy_size,
    ]

    lb = ensure_elastic_label_and_membrane(
        raw_label_path=args.raw_label_path,
        elastic_label_path=args.elastic_label_path,
        elastic_membrane_dir=args.elastic_membrane_dir,
        crop_slices=label_crop,
        point_spacing=tuple(args.point_spacing),
        jitter_sigma=tuple(args.jitter_sigma),
        padding=args.padding,
        membrane_iteration=args.membrane_iteration,
    )

    if has_png_stack(args.output_condition_dir, expected_count=lb.shape[0]):
        print(f"Skip condition masks: {args.output_condition_dir}")
    else:
        print(f"Load mitochondria mask library: {args.mito_mask_dir}")
        mito_img = read_png(args.mito_mask_dir)
        feature_ls, feature_angles, feature_sizes = extract_mask_library_3d(mito_img)
        print(f"Mitochondria templates: {len(feature_ls)}")

        print(f"Generate condition masks: {args.output_condition_dir}")
        lb_to_mask(
            lb,
            args.select_neurons_num,
            args.sample_neurons_num,
            args.output_condition_dir,
            args.elastic_membrane_dir,
            feature_ls,
            feature_angles,
            feature_sizes,
        )
