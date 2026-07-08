# NeuroDiff

[Chinese](README_zh.md)

Official implementation of **Diffusion Model-Based Data Augmentation for Enhanced Neuron Segmentation**.

[[arXiv](https://arxiv.org/abs/2601.15779)] [[Paper PDF](https://arxiv.org/pdf/2601.15779)]

NeuroDiff is a diffusion-based data augmentation framework for neuron segmentation in 3D electron microscopy (EM). It trains a resolution-aware conditional diffusion model to synthesize EM images from 3D biological masks, and uses a biology-guided mask remodeling module to generate structurally plausible image-label pairs for segmentation training.

## Overview

```text
Real EM image + condition mask
        |
        v
Train NeuroDiff conditional diffusion model
        |
        v
Biology-guided mask remodeling
  - membrane elastic deformation
  - mitochondria remodeling
        |
        v
Sample synthetic EM images from remodeled masks
        |
        v
Train neuron segmentation model with generated image-label pairs
```

## Installation

The experiments were conducted with Python 3.10.13, CUDA 11.8, PyTorch 2.1.1, and Mamba.

```bash
conda create -n MD python=3.10.13
conda activate MD

conda install cudatoolkit==11.8 -c nvidia
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
conda install packaging

pip install causal-conv1d==1.1.1
pip install mamba-ssm==1.1.4

pip install torchio==0.18.86 numpy nibabel==4.0.2
pip install einops tensorboard tensorboardX scikit-learn matplotlib h5py scikit-image shapely scipy pillow tqdm PyYAML attrdict opencv-python monai
```

Or install the Python dependencies with:

```bash
pip install -r requirements.txt
```

`waterz` and `elf-segmentation` are needed for segmentation post-processing and evaluation.

## Data Preparation

The scripts expect H5 files to store volumes under the dataset key `main`.

The paper uses the public AC3/AC4 datasets with `6 x 6 x 29 nm^3` resolution.
They can be downloaded from:

- [AC3/AC4 package](https://lichtman.rc.fas.harvard.edu/vast/AC3AC4Package.zip)

After downloading the data, organize or convert the volumes into the H5 layout
below.

Example layout:

```text
data/
  EM/
    AC4-t/
      condition/
      image/
      mito/

segmentation_scripts/
  data/
    AC3AC4/
      AC3_inputs.h5
      AC3_labels.h5
      AC4_inputs.h5
      AC4_labels.h5
    generate_AC/
      AC4_inputs_512.h5
      AC4_labels_512.h5
```

In the experiments reported in the paper, AC4 is the original training volume with
shape `100 x 1024 x 1024`. The 4% training setting uses a cropped subset of this
volume: `16 x 512 x 512`, denoted as `AC4-t` in this repository. The files
`AC4_inputs_512.h5` and `AC4_labels_512.h5` are the center-cropped
`512 x 512` H5 volumes used as the real baseline training data. The folders
`data/EM/AC4-t/condition/` and `data/EM/AC4-t/image/` contain the corresponding
PNG stacks used to train the diffusion model.

You can also pass custom paths through the command-line arguments shown below.

## Usage

### 1. Train NeuroDiff

```bash
python train.py \
  --inputfolder ./data/EM/AC4-t/condition/ \
  --targetfolder ./data/EM/AC4-t/image/ \
  --results_folder ./results/neurodiff_AC4_t/ \
  --gpu 0
```

### 2. Generate remodeled masks

```bash
python tools/mask_remodeling.py \
  --raw_label_path ./segmentation_scripts/data/AC3AC4/AC4_labels.h5 \
  --mito_mask_dir ./data/EM/AC4-t/mito/ \
  --elastic_label_path ./segmentation_scripts/data/generate_AC/AC4_labels-t.h5 \
  --elastic_membrane_dir ./data/EM/AC4-t/affs-ela/ \
  --output_condition_dir ./data/EM/AC4-t/mask-ela/
```

This step produces the remodeled segmentation label `AC4_labels-t.h5` and the condition masks used for diffusion sampling.

### 3. Sample synthetic EM images

```bash
python sample.py \
  --inputfolder ./data/EM/AC4-t/mask-ela/ \
  --exportfolder ./segmentation_scripts/data/generate_AC/mask-t_NeuroDiff/ \
  --weightfile ./results/neurodiff_AC4_t/model-10.pt \
  --gpu 0
```

### 4. Convert generated PNG slices to H5

```bash
python tools/png_stack_to_h5.py \
  --input_dir ./segmentation_scripts/data/generate_AC/mask-t_NeuroDiff/image \
  --output_h5 ./segmentation_scripts/data/generate_AC/mask-t_NeuroDiff.h5 \
  --overwrite
```

### 5. Train segmentation models

```bash
cd segmentation_scripts
```

Train with the original training set:

```bash
python main.py -c example_ac4_4% -m train
```

Train with NeuroDiff-augmented data:

```bash
python main.py -c example_ac4_4%_aug -m train
```

### 6. Inference and evaluation

```bash
python inference.py -c example_ac4_4%_aug -mn example_ac4_4%_aug -m AC3 -ts 100
python evaluate.py -mn example_ac4_4%_aug -m AC3 -ts 100
python get_results.py -mn example_ac4_4%_aug -m AC3
```

## Citation

If this repository is useful for your research, please cite:

```bibtex
@article{jiang2026diffusion,
  title={Diffusion Model-Based Data Augmentation for Enhanced Neuron Segmentation},
  author={Jiang, Liuyun and Zhang, Yanchao and Guo, Jinyue and Lu, Yizhuo and Zhou, Ruining and Han, Hua},
  journal={arXiv preprint arXiv:2601.15779},
  year={2026},
  doi={10.48550/arXiv.2601.15779}
}
```
