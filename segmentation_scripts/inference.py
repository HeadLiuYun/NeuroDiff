import argparse
import os
import time
import warnings
from collections import OrderedDict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
warnings.filterwarnings("ignore")

import cv2
import h5py
import numpy as np
import torch
import torch.nn as nn
import yaml
from attrdict import AttrDict
from tqdm import tqdm

from dataloader.provider_valid import Provider_valid
from model.model_SegMamba import SegMamba
from model.model_Swin_UNETR import SwinUNETR
from model.model_superhuman import UNet_PNI


def parse_args():
    parser = argparse.ArgumentParser(description="Run neuron segmentation inference.")
    parser.add_argument(
        "-c",
        "--cfg",
        type=str,
        default="example_ac4_4%",
        help="Config name without the .yaml suffix.",
    )
    parser.add_argument(
        "-mn",
        "--model_name",
        type=str,
        default=None,
        help="Folder name under checkpoint_dir. Defaults to cfg.TEST.model_name or cfg name.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="AC3",
        choices=["AC3", "AC4"],
        help="Validation/test dataset.",
    )
    parser.add_argument(
        "-ts",
        "--test_split",
        type=int,
        default=100,
        help="Number of z-slices used for inference. Negative values use tail slices.",
    )
    parser.add_argument(
        "--model_ids",
        type=str,
        default="model-200000",
        help="Comma-separated checkpoint ids without .ckpt.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./trained_model",
        help="Root folder containing trained model checkpoints.",
    )
    parser.add_argument("-sw", "--show", action="store_true", help="Save affinity preview PNGs.")
    return parser.parse_args()


def load_config(config_name):
    cfg_file = config_name + ".yaml"
    with open(os.path.join("./config", cfg_file), "r") as f:
        return AttrDict(yaml.load(f, Loader=yaml.Loader))


def build_model(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_type = cfg.MODEL.model_type

    if model_type == "superhuman":
        model = UNet_PNI(
            in_planes=cfg.MODEL.input_nc,
            out_planes=cfg.MODEL.output_nc,
            filters=cfg.MODEL.filters,
            upsample_mode=cfg.MODEL.upsample_mode,
            decode_ratio=cfg.MODEL.decode_ratio,
            merge_mode=cfg.MODEL.merge_mode,
            pad_mode=cfg.MODEL.pad_mode,
            bn_mode=cfg.MODEL.bn_mode,
            relu_mode=cfg.MODEL.relu_mode,
            init_mode=cfg.MODEL.init_mode,
            show_feature=False,
        ).to(device)
    elif model_type == "SwinUNETR":
        model = SwinUNETR(
            img_size=(64, 96, 96),
            in_channels=cfg.MODEL.input_nc,
            out_channels=cfg.MODEL.output_nc,
            feature_size=48,
        ).to(device)
    elif model_type == "SegMamba":
        model = SegMamba(
            in_chans=cfg.MODEL.input_nc,
            out_chans=cfg.MODEL.output_nc,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            hidden_size=768,
        ).to(device)
    else:
        raise ValueError("Unsupported model_type: %s" % model_type)

    return model, device


def build_criterion(cfg):
    if cfg.TRAIN.loss_func == "MSELoss":
        return nn.MSELoss()
    if cfg.TRAIN.loss_func == "BCELoss":
        return nn.BCELoss()
    raise ValueError("Unsupported loss function: %s" % cfg.TRAIN.loss_func)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_weights"]
    cleaned_state_dict = OrderedDict()
    for key, value in state_dict.items():
        cleaned_key = key[7:] if key.startswith("module.") else key
        cleaned_state_dict[cleaned_key] = value
    model.load_state_dict(cleaned_state_dict)
    return model


def save_affinity_preview(output_affs, gt_affs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_affs = (output_affs * 255).astype(np.uint8)
    gt_affs = (gt_affs * 255).astype(np.uint8)
    for i in range(output_affs.shape[1]):
        preview = np.concatenate(
            [output_affs[0, i], output_affs[1, i], output_affs[2, i]],
            axis=1,
        )
        cv2.imwrite(os.path.join(output_dir, str(i).zfill(4) + ".png"), preview)


def run_single_checkpoint(cfg, args, trained_model, model_id):
    out_affs = os.path.join("./inference", trained_model, args.mode, "affs_" + model_id)
    os.makedirs(out_affs, exist_ok=True)
    print("out_path:", out_affs)

    model, device = build_model(cfg)
    ckpt_path = os.path.join(args.checkpoint_dir, trained_model, model_id + ".ckpt")
    model = load_checkpoint(model, ckpt_path, device)
    model.eval()

    valid_provider = Provider_valid(
        cfg,
        valid_data=args.mode,
        test_split=args.test_split,
        test=True,
    )
    val_loader = torch.utils.data.DataLoader(valid_provider, batch_size=1)
    criterion = build_criterion(cfg)

    losses_valid = []
    start = time.time()
    pbar = tqdm(total=len(valid_provider))
    for data in val_loader:
        inputs, target, _ = data
        inputs = inputs.to(device)
        target = target[:, :3].to(device)
        with torch.no_grad():
            pred = model(inputs)[:, :3]

        losses_valid.append(criterion(pred, target).item())
        valid_provider.add_vol(np.squeeze(pred.data.cpu().numpy()))
        pbar.update(1)
    pbar.close()

    cost_time = time.time() - start
    output_affs = valid_provider.get_results()
    gt_affs = valid_provider.get_gt_affs()
    valid_provider.reset_output()

    with open(os.path.join(out_affs, "scores.txt"), "w") as f:
        f.write("Inference time=%.6f\n" % cost_time)
        f.write("epoch_loss=%.6f\n" % (sum(losses_valid) / len(losses_valid)))

    print("Inference time=%.6f" % cost_time)
    print("save affs...")
    print("the shape of affs:", output_affs.shape)
    with h5py.File(os.path.join(out_affs, "affs.h5"), "w") as f:
        f.create_dataset("main", data=output_affs, dtype=np.float32, compression="gzip")

    if args.show:
        save_affinity_preview(output_affs, gt_affs, os.path.join(out_affs, "affs_img"))
    print("Done")


def main():
    args = parse_args()
    cfg = load_config(args.cfg)
    trained_model = args.model_name or cfg.TEST.model_name or args.cfg
    model_ids = [item.strip() for item in args.model_ids.split(",") if item.strip()]

    print("cfg_file:", args.cfg + ".yaml")
    print("trained_model:", trained_model)
    print("model_ids:", model_ids)

    for model_id in model_ids:
        run_single_checkpoint(cfg, args, trained_model, model_id)


if __name__ == "__main__":
    main()
