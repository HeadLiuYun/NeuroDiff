import argparse
import os

import torch
from torchvision.transforms import Compose, Lambda

from dataset import NiftiImageGenerator, NiftiPairImageGenerator
from diffusion_model.neurodiff import create_neurodiff
from diffusion_model.trainer import GaussianDiffusion, Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train NeuroDiff.")
    parser.add_argument("-i", "--inputfolder", type=str, default="./data/EM/AC4-t/condition/")
    parser.add_argument("-t", "--targetfolder", type=str, default="./data/EM/AC4-t/image/")
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--depth_size", type=int, default=8)
    parser.add_argument("--num_channels", type=int, default=64)
    parser.add_argument("--num_res_blocks", type=int, default=1)
    parser.add_argument("--num_class_labels", type=int, default=3)
    parser.add_argument("--train_lr", type=float, default=1e-5)
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--timesteps", type=int, default=250)
    parser.add_argument("--save_and_sample_every", type=int, default=1000)
    parser.add_argument("--with_condition", dest="with_condition", action="store_true")
    parser.add_argument("--no_condition", dest="with_condition", action="store_false")
    parser.set_defaults(with_condition=True)
    parser.add_argument("-r", "--resume_weight", type=str, default="")
    parser.add_argument("--results_folder", type=str, default="./results/neurodiff_AC4_t/")
    parser.add_argument("--res_z", type=int, default=29)
    parser.add_argument("--res_xy", type=int, default=6)
    parser.add_argument("--gpu", type=str, default="0", help="CUDA device id, for example: 0 or 1.")
    return parser.parse_args()


def build_transforms():
    target_transform = Compose([
        Lambda(lambda t: torch.tensor(t).float()),
        Lambda(lambda t: (t * 2) - 1),
        Lambda(lambda t: t.unsqueeze(0)),
        Lambda(lambda t: t.transpose(3, 1)),
    ])

    condition_transform = Compose([
        Lambda(lambda t: torch.tensor(t).float()),
        Lambda(lambda t: (t * 2) - 1),
        Lambda(lambda t: t.permute(3, 0, 1, 2)),
        Lambda(lambda t: t.transpose(3, 1)),
    ])
    return condition_transform, target_transform


def build_dataset(args, condition_transform, target_transform):
    if args.with_condition:
        return NiftiPairImageGenerator(
            args.inputfolder,
            args.targetfolder,
            input_size=args.input_size,
            depth_size=args.depth_size,
            transform=condition_transform,
            target_transform=target_transform,
            full_channel_mask=True,
        )

    return NiftiImageGenerator(
        args.inputfolder,
        input_size=args.input_size,
        depth_size=args.depth_size,
        transform=target_transform,
    )


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    condition_transform, target_transform = build_transforms()
    dataset = build_dataset(args, condition_transform, target_transform)

    in_channels = args.num_class_labels if args.with_condition else 1
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
        with_condition=args.with_condition,
        channels=out_channels,
    ).cuda()

    if args.resume_weight:
        weight = torch.load(args.resume_weight, map_location="cuda")
        diffusion.load_state_dict(weight["ema"])
        print("Model loaded.")

    trainer = Trainer(
        diffusion,
        dataset,
        image_size=args.input_size,
        depth_size=args.depth_size,
        train_batch_size=args.batchsize,
        train_lr=args.train_lr,
        train_num_steps=args.epochs,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        fp16=False,
        with_condition=args.with_condition,
        save_and_sample_every=args.save_and_sample_every,
        results_folder=args.results_folder,
    )

    trainer.train()


if __name__ == "__main__":
    main()
