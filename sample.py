import argparse
import os
import time

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Lambda

from diffusion_model.neurodiff import create_neurodiff
from diffusion_model.trainer import GaussianDiffusion


def parse_args():
    parser = argparse.ArgumentParser(description="Generate EM images with NeuroDiff.")
    parser.add_argument(
        "-i",
        "--inputfolder",
        type=str,
        default="./data/EM/AC4-t/mask-ela/",
        help="Folder containing condition mask PNGs.",
    )
    parser.add_argument(
        "-e",
        "--exportfolder",
        type=str,
        default="./segmentation_scripts/data/generate_AC/mask-t_NeuroDiff/",
        help="Output folder. Generated images are saved to image/ and copied masks to mask/.",
    )
    parser.add_argument("-w", "--weightfile", type=str, default="./results/neurodiff_AC4_t/model-10.pt")
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--depth_size", type=int, default=8)
    parser.add_argument("--input_depth", type=int, default=16)
    parser.add_argument("--num_channels", type=int, default=64)
    parser.add_argument("--num_res_blocks", type=int, default=1)
    parser.add_argument("--num_class_labels", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=250)
    parser.add_argument("--res_z", type=int, default=29)
    parser.add_argument("--res_xy", type=int, default=6)
    parser.add_argument("--stride_z", type=int, default=None)
    parser.add_argument("--stride_xy", type=int, default=None)
    parser.add_argument("--gpu", type=str, default="0", help="CUDA device id, for example: 0 or 1.")
    return parser.parse_args()


def read_mask_stack(folder, input_depth):
    png_files = sorted(
        [f for f in os.listdir(folder) if f.endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in {folder}")

    img_list = []
    for file_name in png_files:
        img = Image.open(os.path.join(folder, file_name)).convert("L")
        img_list.append(np.array(img))

    img = np.stack(img_list, axis=0)
    img = np.transpose(img, (1, 2, 0))
    return img[:, :, :input_depth]


def save_png_stack(volume, output_dir, input_depth, bias=False, start_index=0):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(volume.shape[0]):
        if i + start_index >= input_depth:
            break

        image_data = volume[i]
        if bias:
            image_data = (image_data + 1) / 2
            image_data = np.clip(image_data, 0, 1)
            image_data = np.uint8(image_data * 255)

        img = Image.fromarray(image_data)
        img.save(os.path.join(output_dir, f"{i + start_index:04d}.png"))

    print(f"Images saved to {output_dir}")


def label_to_condition(mask_volume):
    condition = np.zeros(mask_volume.shape + (2,), dtype=np.float32)
    condition[mask_volume == 1, 0] = 1
    condition[mask_volume == 2, 1] = 1
    return condition


def build_input_transform():
    return Compose([
        Lambda(lambda t: torch.tensor(t).float()),
        Lambda(lambda t: (t * 2) - 1),
        Lambda(lambda t: t.permute(3, 0, 1, 2)),
        Lambda(lambda t: t.unsqueeze(0)),
        Lambda(lambda t: t.transpose(4, 2)),
    ])


def uniform_window_3d(window_size):
    window = np.ones(window_size, dtype=np.float32)
    window /= window.sum()
    return window


def build_diffusion(args):
    in_channels = args.num_class_labels
    out_channels = 1

    model = create_neurodiff(
        args.input_size,
        args.num_channels,
        args.num_res_blocks,
        in_channels=in_channels,
        out_channels=out_channels,
        ani_ratio=args.input_size / args.depth_size,
        res_z=args.res_z,
        res_xy=args.res_xy,
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=args.input_size,
        depth_size=args.depth_size,
        timesteps=args.timesteps,
        loss_type="l1",
        with_condition=True,
        channels=out_channels,
    ).cuda()

    checkpoint = torch.load(args.weightfile, map_location="cuda")
    diffusion.load_state_dict(checkpoint["ema"])
    diffusion.eval()
    print(f"Model loaded: {args.weightfile}")
    return diffusion


@torch.no_grad()
def sample_volume(args, diffusion, input_tensor):
    stride_z = args.stride_z or args.depth_size
    stride_xy = args.stride_xy or args.input_size
    sub_tensor_size = (args.depth_size, args.input_size, args.input_size)

    output_volume = torch.zeros((input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4])).cuda()
    weight_volume = torch.zeros_like(output_volume)
    window_weight = torch.tensor(uniform_window_3d(sub_tensor_size)).cuda()

    for z in range(0, input_tensor.shape[2], stride_z):
        for y in range(0, input_tensor.shape[3], stride_xy):
            for x in range(0, input_tensor.shape[4], stride_xy):
                z_end = min(z + sub_tensor_size[0], input_tensor.shape[2])
                y_end = min(y + sub_tensor_size[1], input_tensor.shape[3])
                x_end = min(x + sub_tensor_size[2], input_tensor.shape[4])

                z_start = max(z_end - sub_tensor_size[0], 0) if z_end != z + sub_tensor_size[0] else z
                y_start = max(y_end - sub_tensor_size[1], 0) if y_end != y + sub_tensor_size[1] else y
                x_start = max(x_end - sub_tensor_size[2], 0) if x_end != x + sub_tensor_size[2] else x

                sub_tensor = input_tensor[
                    :,
                    :,
                    z_start:z_start + sub_tensor_size[0],
                    y_start:y_start + sub_tensor_size[1],
                    x_start:x_start + sub_tensor_size[2],
                ].cuda()

                sample = diffusion.sample(batch_size=1, condition_tensors=sub_tensor)
                patch = sample[0, 0].reshape(args.depth_size, args.input_size, args.input_size)

                output_volume[
                    z_start:z_start + sub_tensor_size[0],
                    y_start:y_start + sub_tensor_size[1],
                    x_start:x_start + sub_tensor_size[2],
                ] += patch * window_weight
                weight_volume[
                    z_start:z_start + sub_tensor_size[0],
                    y_start:y_start + sub_tensor_size[1],
                    x_start:x_start + sub_tensor_size[2],
                ] += window_weight

    return output_volume / weight_volume


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    start_time = time.time()
    img_dir = os.path.join(args.exportfolder, "image")
    mask_dir = os.path.join(args.exportfolder, "mask")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    mask_volume = read_mask_stack(args.inputfolder, args.input_depth)
    condition = label_to_condition(mask_volume)
    input_tensor = build_input_transform()(condition)
    print(input_tensor.shape)

    diffusion = build_diffusion(args)
    final_output = sample_volume(args, diffusion, input_tensor)

    final_output = final_output.cpu().numpy()
    final_output = np.transpose(final_output, (0, 2, 1))
    save_png_stack(final_output, img_dir, args.input_depth, bias=True)

    mask_volume = np.transpose(mask_volume, (2, 0, 1))
    save_png_stack(mask_volume, mask_dir, args.input_depth)

    torch.cuda.empty_cache()
    elapsed_minutes = (time.time() - start_time) / 60
    print(f"Done. Total execution time: {elapsed_minutes:.2f} minutes")


if __name__ == "__main__":
    main()
