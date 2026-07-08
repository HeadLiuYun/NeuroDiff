# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a folder of 2D PNG slices into one H5 volume."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="segmentation_scripts/data/generate_AC/mask-t_NeuroDiff/image",
        help="Folder containing generated PNG slices.",
    )
    parser.add_argument(
        "--output_h5",
        type=str,
        default="segmentation_scripts/data/generate_AC/mask-t_NeuroDiff.h5",
        help="Output H5 file used by the segmentation dataloader.",
    )
    parser.add_argument(
        "--dataset_key",
        type=str,
        default="main",
        help="Dataset name inside the H5 file.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Optional first slice index after sorting.",
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=None,
        help="Optional stop slice index after sorting.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output_h5 if it already exists.",
    )
    return parser.parse_args()


def numeric_sort_key(path):
    stem = Path(path).stem
    return int(stem) if stem.isdigit() else stem


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def read_png_stack(input_dir, start=None, stop=None):
    input_path = resolve_path(input_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_path}")

    png_files = sorted(input_path.glob("*.png"), key=numeric_sort_key)
    png_files = png_files[start:stop]

    if not png_files:
        raise FileNotFoundError(f"No PNG files found in: {input_path}")

    slices = []
    for png_file in png_files:
        image = Image.open(png_file).convert("L")
        slices.append(np.asarray(image))

    return np.stack(slices, axis=0)


def write_h5(output_h5, data, dataset_key="main", overwrite=False):
    output_path = resolve_path(output_h5)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    with h5py.File(output_path, "w") as f:
        f.create_dataset(
            dataset_key,
            data=data,
            dtype=data.dtype,
            compression="gzip",
        )


def main():
    args = parse_args()

    data = read_png_stack(args.input_dir, start=args.start, stop=args.stop)
    print(f"Input folder: {resolve_path(args.input_dir)}")
    print(f"Output H5: {resolve_path(args.output_h5)}")
    print(f"Volume shape: {data.shape}")
    print(f"Volume dtype: {data.dtype}")

    write_h5(
        args.output_h5,
        data,
        dataset_key=args.dataset_key,
        overwrite=args.overwrite,
    )
    print("Done.")


if __name__ == "__main__":
    main()
