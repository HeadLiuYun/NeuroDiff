from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
import torch.nn as nn
import yaml
from attrdict import AttrDict
from tensorboardX import SummaryWriter

from dataloader.provider_train import Provider
from model.model_SegMamba import SegMamba
from model.model_Swin_UNETR import SwinUNETR
from model.model_superhuman import UNet_PNI
from utils.utils import setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a neuron segmentation model.")
    parser.add_argument(
        "-c",
        "--cfg",
        type=str,
        default="example_ac4_4%",
        help="Config name without the .yaml suffix.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="train",
        choices=["train"],
        help="Run mode.",
    )
    return parser.parse_args()


def load_config(config_name):
    cfg_file = config_name + ".yaml"
    with open(os.path.join("./config", cfg_file), "r") as f:
        cfg = AttrDict(yaml.load(f, Loader=yaml.Loader))
    cfg.path = cfg_file
    cfg.time = time.strftime("%Y-%m-%d--%H-%M-%S", time.localtime())
    return cfg


def init_project(cfg):
    setup_seed(cfg.TRAIN.random_seed)
    if cfg.TRAIN.if_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is requested, but no GPU is available.")

    model_name = cfg.time + "_" + cfg.NAME
    cfg.cache_path = os.path.join(cfg.TRAIN.cache_path, model_name)
    cfg.save_path = os.path.join(cfg.TRAIN.save_path, model_name)
    cfg.record_path = os.path.join(cfg.save_path, model_name)
    cfg.valid_path = os.path.join(cfg.save_path, "valid")

    for path in [cfg.cache_path, cfg.save_path, cfg.record_path, cfg.valid_path]:
        os.makedirs(path, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="%m-%d %H:%M",
        filename=os.path.join(cfg.record_path, cfg.time + ".log"),
        filemode="w",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("").addHandler(console)

    logging.info(cfg)
    writer = SummaryWriter(cfg.record_path)
    writer.add_text("cfg", str(cfg))
    return writer


def load_dataset(cfg):
    print("Caching datasets ... ", flush=True)
    start = time.time()
    train_provider = Provider("train", cfg)
    print("Done (time: %.2fs)" % (time.time() - start))
    return train_provider


def build_model(cfg):
    print("Building model ... ", end="", flush=True)
    start = time.time()
    device = torch.device("cuda:0" if cfg.TRAIN.if_cuda else "cpu")
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

    cuda_count = torch.cuda.device_count() if cfg.TRAIN.if_cuda else 0
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count != 0:
            raise ValueError(
                "Batch size (%d) cannot be divided by GPU number (%d)."
                % (cfg.TRAIN.batch_size, cuda_count)
            )
        print("%d GPUs ... " % cuda_count, end="", flush=True)
        model = nn.DataParallel(model)
    else:
        print("single GPU/CPU ... ", end="", flush=True)

    print("Done (time: %.2fs)" % (time.time() - start))
    return model


def resume_params(cfg, model, optimizer):
    if not cfg.TRAIN.resume:
        return model, optimizer, 0

    model_path = cfg.TRAIN.resume_path
    print("Resuming weights from %s ... " % model_path, end="", flush=True)
    if not os.path.isfile(model_path):
        raise FileNotFoundError("No checkpoint found at %s" % model_path)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_weights"])
    print("Done.")
    return model, optimizer, checkpoint["current_iter"]


def calculate_lr(cfg, iters):
    if cfg.TRAIN.end_lr == cfg.TRAIN.base_lr:
        return cfg.TRAIN.base_lr
    if iters < cfg.TRAIN.warmup_iters:
        return (
            (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr)
            * pow(float(iters) / cfg.TRAIN.warmup_iters, cfg.TRAIN.power)
            + cfg.TRAIN.end_lr
        )
    if iters < cfg.TRAIN.decay_iters:
        return (
            (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr)
            * pow(1 - float(iters - cfg.TRAIN.warmup_iters) / cfg.TRAIN.decay_iters, cfg.TRAIN.power)
            + cfg.TRAIN.end_lr
        )
    return cfg.TRAIN.end_lr


def build_criterion(cfg):
    if cfg.TRAIN.loss_func == "MSELoss":
        return nn.MSELoss()
    if cfg.TRAIN.loss_func == "BCELoss":
        return nn.BCELoss()
    raise ValueError("Unsupported loss function: %s" % cfg.TRAIN.loss_func)


def train_loop(cfg, train_provider, model, optimizer, iters, writer):
    criterion = build_criterion(cfg)
    loss_file = open(os.path.join(cfg.record_path, "loss.txt"), "a")
    sum_time = 0
    sum_loss = 0
    recent_times = []

    while iters <= cfg.TRAIN.total_iters:
        model.train()
        iters += 1
        start = time.time()

        inputs, target, _ = train_provider.next()
        current_lr = calculate_lr(cfg, iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        optimizer.zero_grad()
        pred = model(inputs)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        sum_time += time.time() - start

        if iters % cfg.TRAIN.display_freq == 0 or iters == 1:
            recent_times.append(sum_time)
            mean_loss = sum_loss if iters == 1 else sum_loss / cfg.TRAIN.display_freq
            remaining_minutes = (
                (cfg.TRAIN.total_iters - iters)
                / cfg.TRAIN.display_freq
                * np.mean(np.asarray(recent_times))
                / 60
            )
            logging.info(
                "step %d, loss = %.6f, lr: %.8f, et: %.2f sec, rd: %.2f min"
                % (iters, mean_loss, current_lr, sum_time, remaining_minutes)
            )
            writer.add_scalar("loss", mean_loss, iters)
            loss_file.write("step = %d, loss = %.12f,\n" % (iters, mean_loss))
            loss_file.flush()
            sys.stdout.flush()
            sum_time = 0
            sum_loss = 0

        if iters % cfg.TRAIN.save_freq == 0 and iters >= cfg.TRAIN.min_save_iters:
            states = {"current_iter": iters, "valid_result": None, "model_weights": model.state_dict()}
            torch.save(states, os.path.join(cfg.save_path, "model-%06d.ckpt" % iters))
            print("Saved model at iteration %d." % iters, flush=True)

    loss_file.close()


def main():
    args = parse_args()
    cfg = load_config(args.cfg)
    print("cfg_file:", cfg.path)
    print("mode:", args.mode)
    print("time stamp:", cfg.time)

    writer = init_project(cfg)
    train_provider = load_dataset(cfg)
    model = build_model(cfg)
    weight_decay = 1e-6 if cfg.TRAIN.weight_decay is None else cfg.TRAIN.weight_decay
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.TRAIN.base_lr,
        betas=(0.9, 0.999),
        eps=0.01,
        weight_decay=weight_decay,
        amsgrad=True,
    )
    model, optimizer, init_iters = resume_params(cfg, model, optimizer)
    train_loop(cfg, train_provider, model, optimizer, init_iters, writer)
    writer.close()
    print("***Done***")


if __name__ == "__main__":
    main()
