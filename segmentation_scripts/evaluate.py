# -*- coding: utf-8 -*-
import argparse
import os

import elf.segmentation.features as feats
import elf.segmentation.multicut as mc
import elf.segmentation.watershed as ws
import h5py
import numpy as np
import waterz
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref


LABEL_FILES = {
    "AC3": ("AC3AC4", "AC3_labels.h5"),
    "AC4": ("AC3AC4", "AC4_labels.h5"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predicted affinities.")
    parser.add_argument("-mn", "--model_name", type=str, default="example_ac4_4%")
    parser.add_argument("-m", "--mode", type=str, default="AC3", choices=sorted(LABEL_FILES))
    parser.add_argument("-ts", "--test_split", type=int, default=100)
    parser.add_argument("--model_id", type=str, default="model-200000")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--inference_dir", type=str, default="./inference")
    parser.add_argument("--no_waterz", action="store_true", help="Skip Waterz post-processing.")
    parser.add_argument("--no_lmc", action="store_true", help="Skip multicut post-processing.")
    return parser.parse_args()


def relabel(seg):
    uid = np.unique(seg)
    if len(uid) == 1 and uid[0] == 0:
        return seg

    uid = uid[uid > 0]
    mapping = np.zeros(int(uid.max()) + 1, dtype=seg.dtype)
    mapping[uid] = np.arange(1, len(uid) + 1, dtype=seg.dtype)
    return mapping[seg]


def load_h5(path):
    with h5py.File(path, "r") as f:
        return f["main"][:]


def write_h5(path, data):
    with h5py.File(path, "w") as f:
        f.create_dataset("main", data=data, dtype=data.dtype, compression="gzip")


def load_gt_label(args):
    sub_dir, label_name = LABEL_FILES[args.mode]
    data = load_h5(os.path.join(args.data_dir, sub_dir, label_name))
    if args.test_split > 0:
        return data[: args.test_split]
    return data[args.test_split :]


def build_watershed(affs):
    boundary_input = np.maximum(affs[1], affs[2])
    fragments = np.zeros_like(boundary_input, dtype="uint64")
    offset = 0
    for z in range(fragments.shape[0]):
        fragment_z, max_id = ws.distance_transform_watershed(
            boundary_input[z],
            threshold=0.25,
            sigma_seeds=2.0,
        )
        fragment_z += offset
        offset += max_id
        fragments[z] = fragment_z
    return fragments, boundary_input


def post_lmc(affs):
    affs = 1 - affs
    fragments, boundary_input = build_watershed(affs)
    rag = feats.compute_rag(fragments)
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    costs = feats.compute_affinity_features(rag, affs, offsets)[:, 0]
    edge_sizes = feats.compute_boundary_mean_and_length(rag, boundary_input)[:, 1]
    costs = mc.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes, beta=0.25)
    node_labels = mc.multicut_kernighan_lin(rag, costs)
    return feats.project_node_labels_to_pixels(rag, node_labels)


def post_waterz(affs):
    affs = 1 - affs
    fragments, _ = build_watershed(affs)
    scoring_function = "OneMinus<HistogramQuantileAffinity<RegionGraphType,50,ScoreValue,256>>"
    generator = waterz.agglomerate(
        1 - affs.copy(),
        [0.30],
        fragments=fragments.copy(),
        return_merge_history=False,
        scoring_function=scoring_function,
        discretize_queue=256,
    )
    return list(generator)[0]


def evaluate_segmentation(name, segmentation, gt_seg, out_dir, score_file):
    segmentation = relabel(segmentation).astype(np.uint64)
    write_h5(os.path.join(out_dir, f"seg_{name}.h5"), segmentation)

    arand = adapted_rand_ref(gt_seg, segmentation, ignore_labels=(0,))[0]
    voi_split, voi_merge = voi_ref(gt_seg, segmentation, ignore_labels=(0,))
    voi_sum = voi_split + voi_merge
    line = (
        f"{name}: voi_split={voi_split:.6f}, voi_merge={voi_merge:.6f}, "
        f"voi_sum={voi_sum:.6f}, arand={arand:.6f}"
    )
    print(line)
    score_file.write(line + "\n")


def main():
    args = parse_args()
    out_dir = os.path.join(
        args.inference_dir,
        args.model_name,
        args.mode,
        "affs_" + args.model_id,
    )
    os.makedirs(out_dir, exist_ok=True)
    print("out_path:", out_dir)

    gt_seg = load_gt_label(args)
    output_affs = load_h5(os.path.join(out_dir, "affs.h5"))[:3]

    with open(os.path.join(out_dir, "scores_post.txt"), "w") as score_file:
        if not args.no_waterz:
            print("Waterz segmentation...")
            evaluate_segmentation("waterz", post_waterz(output_affs), gt_seg, out_dir, score_file)

        if not args.no_lmc:
            print("LMC segmentation...")
            evaluate_segmentation("lmc", post_lmc(output_affs), gt_seg, out_dir, score_file)


if __name__ == "__main__":
    main()
