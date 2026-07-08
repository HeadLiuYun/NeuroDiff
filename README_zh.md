# NeuroDiff

[English](README.md)

本文是 **Diffusion Model-Based Data Augmentation for Enhanced Neuron Segmentation** 的官方代码实现，论文已发表于 **IEEE ISBI 2026**。

[[IEEE](https://ieeexplore.ieee.org/document/11515854)] [[arXiv](https://arxiv.org/abs/2601.15779)] [[Paper PDF](https://arxiv.org/pdf/2601.15779)]

NeuroDiff 是一个面向 3D 电镜（EM）神经元分割的数据增强框架。它先训练一个 resolution-aware 条件扩散模型，根据 3D 生物结构 mask 合成 EM 图像；再通过 biology-guided mask remodeling 生成结构更丰富、更加合理的 image-label pair，用于提升神经元分割模型的训练效果。

## 方法流程

```text
真实 EM 图像 + condition mask
        |
        v
训练 NeuroDiff 条件扩散模型
        |
        v
Biology-guided mask remodeling
  - membrane elastic deformation
  - mitochondria remodeling
        |
        v
根据 remodeled mask 生成新的 EM 图像
        |
        v
使用生成的 image-label pair 训练神经元分割模型
```

## 安装环境

实验环境使用 Python 3.10.13、CUDA 11.8、PyTorch 2.1.1 和 Mamba。

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

或者直接安装：

```bash
pip install -r requirements.txt
```

分割后处理和评估需要 `waterz` 与 `elf-segmentation`。

## 数据准备

H5 文件默认从 key 为 `main` 的 dataset 中读取数据。

论文中使用的是公开的 AC3/AC4 数据集，分辨率为 `6 x 6 x 29 nm^3`。可以从下面链接下载：

- [AC3/AC4 package](https://lichtman.rc.fas.harvard.edu/vast/AC3AC4Package.zip)

下载后，将数据整理或转换成下面的 H5 文件结构。

推荐的数据结构如下：

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

在论文实验中，AC4 原始训练体数据大小为 `100 x 1024 x 1024`。其中 4%
训练数据设置对应截取出的 `16 x 512 x 512` 子体数据，在本仓库中记为
`AC4-t`。`AC4_inputs_512.h5` 和 `AC4_labels_512.h5` 对应的是从 AC4
中间区域裁剪出的 `512 x 512` H5 数据，用作真实数据 baseline 的训练集。
`data/EM/AC4-t/condition/` 和 `data/EM/AC4-t/image/` 则是训练扩散模型时
使用的对应 PNG stack。

也可以在运行命令时通过参数传入自己的数据路径。

## 使用流程

### 1. 训练 NeuroDiff

```bash
python train.py \
  --inputfolder ./data/EM/AC4-t/condition/ \
  --targetfolder ./data/EM/AC4-t/image/ \
  --results_folder ./results/neurodiff_AC4_t/ \
  --gpu 0
```

### 2. 生成 remodeled mask

```bash
python tools/mask_remodeling.py \
  --raw_label_path ./segmentation_scripts/data/AC3AC4/AC4_labels.h5 \
  --mito_mask_dir ./data/EM/AC4-t/mito/ \
  --elastic_label_path ./segmentation_scripts/data/generate_AC/AC4_labels-t.h5 \
  --elastic_membrane_dir ./data/EM/AC4-t/affs-ela/ \
  --output_condition_dir ./data/EM/AC4-t/mask-ela/
```

这一步会生成分割训练需要的新标签 `AC4_labels-t.h5`，以及扩散模型采样需要的 condition mask。

### 3. 生成新的 EM 图像

```bash
python sample.py \
  --inputfolder ./data/EM/AC4-t/mask-ela/ \
  --exportfolder ./segmentation_scripts/data/generate_AC/mask-t_NeuroDiff/ \
  --weightfile ./results/neurodiff_AC4_t/model-10.pt \
  --gpu 0
```

### 4. 将生成的 PNG 切片转成 H5

```bash
python tools/png_stack_to_h5.py \
  --input_dir ./segmentation_scripts/data/generate_AC/mask-t_NeuroDiff/image \
  --output_h5 ./segmentation_scripts/data/generate_AC/mask-t_NeuroDiff.h5 \
  --overwrite
```

### 5. 训练神经元分割模型

```bash
cd segmentation_scripts
```

只使用原始训练集：

```bash
python main.py -c example_ac4_4% -m train
```

使用 NeuroDiff 增强数据：

```bash
python main.py -c example_ac4_4%_aug -m train
```

### 6. 推理和评估

```bash
python inference.py -c example_ac4_4%_aug -mn example_ac4_4%_aug -m AC3 -ts 100
python evaluate.py -mn example_ac4_4%_aug -m AC3 -ts 100
python get_results.py -mn example_ac4_4%_aug -m AC3
```

## Citation

如果本项目对你的研究有帮助，请引用：

```bibtex
@INPROCEEDINGS{11515854,
  author={Jiang, Liuyun and Zhang, Yanchao and Guo, Jinyue and Lu, Yizhuo and Zhou, Ruining and Han, Hua},
  booktitle={2026 IEEE 23rd International Symposium on Biomedical Imaging (ISBI)},
  title={Diffusion Model-Based Data Augmentation for Enhanced Neuron Segmentation},
  year={2026},
  pages={01-05},
  doi={10.1109/ISBI61048.2026.11515854}
}
```
