# -*- coding:utf-8 -*-
# *Main part of the code is adopted from the following repository: https://github.com/openai/guided-diffusion

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .modules import *

NUM_CLASSES = 1


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class MambaLayer(nn.Module):
    def __init__(self, dim, sequence, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.sequence = sequence
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )

    def forward(self, x):
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        # transpose
        x = x.permute(self.sequence)
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        reverse_sequence = []
        for i in range(len(self.sequence)):
            reverse_sequence.append(self.sequence.index(i))
        out = out.permute(reverse_sequence)
        return out


def get_positional_encoding(shape, mlp, device, res_z=10, res_xy=5):
    D, W, H = shape
    z, y, x = th.meshgrid(
        th.arange(D, device=device),
        th.arange(W, device=device),
        th.arange(H, device=device),
        indexing='ij'
    )

    z = z * res_z
    y = y * res_xy
    x = x * res_xy

    coords = th.stack([z, y, x], dim=-1).float()  # [D, W, H, 3]
    coords = coords.reshape(-1, 3)  # [D*W*H, 3]

    pe = mlp(coords)
    pe = pe.transpose(0, 1).reshape(1, -1, D, W, H)

    return pe


class PosMLP(nn.Module):
    def __init__(self, in_dim=3, embed_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, coords):
        return self.mlp(coords)


class CrossMamba(nn.Module):
    def __init__(self, in_planes, res_z, res_xy):
        super(CrossMamba, self).__init__()

        self.block1 = MambaLayer(in_planes, [0, 1, 2, 3, 4])  # x→y→z
        # self.block2 = MambaLayer(in_planes, [0, 1, 2, 4, 3])  # y→x→z
        # self.block3 = MambaLayer(in_planes, [0, 1, 3, 4, 2])  # z→x→y
        # self.block4 = MambaLayer(in_planes, [0, 1, 4, 3, 2])  # z→y→x
        self.pos_mlp = PosMLP(in_dim=3, embed_dim=in_planes)
        self.res_z = res_z
        self.res_xy = res_xy

    def forward(self, x):
        B, C, D, W, H = x.shape
        pos_encoding = get_positional_encoding((D, W, H), self.pos_mlp, x.device, self.res_z,
                                               self.res_xy)  # [1, C, D, W, H]

        x = x + pos_encoding  # Broadcasting over batch

        out = self.block1(x)
        # out2 = self.block2(x)

        return out


# MambaLayer(in_planes, [0, 1, 2, 3, 4])  # x→y→z

class NeuroDiff(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=3,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            ani_ratio=1,
            res_z=10,
            res_xy=5,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.ani_ratio = ani_ratio

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )

        self.cond_pre = nn.Sequential(
            conv_nd(3, 2, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(3, 32, 64, 3, padding=1, stride=(1, 2, 2)),
            nn.SiLU(),
            conv_nd(3, 64, 128, 3, padding=1, stride=(1, 2, 2)),
            nn.SiLU(),
            conv_nd(3, 128, 256, 3, padding=1, stride=(1, 2, 2)),
        )
        pre_ratio = (1, 8, 8)
        dense_ch = 256
        self.dense_hint_blocks = nn.ModuleList([])
        current_ratio = (1, 1, 1)

        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                self.dense_hint_blocks.append(
                    self.make_dense_hint_block(dense_ch, int(mult * model_channels), current_ratio, pre_ratio))

                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        # MambaLayer(ch, [0, 1, 2, 3, 4])  # x→y→z
                        CrossMamba(ch, res_z, res_xy)
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch, ani_stride=True
                        ) if ds < self.ani_ratio else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )

                if ds < self.ani_ratio:
                    stride = (1, 2, 2)
                else:
                    stride = (2, 2, 2)
                current_ratio = (
                    current_ratio[0] * stride[0],
                    current_ratio[1] * stride[1],
                    current_ratio[2] * stride[2]
                )

                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            CrossMamba(ch, res_z, res_xy),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        CrossMamba(ch, res_z, res_xy)
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch,
                                      ani_stride=True) if ds <= self.ani_ratio else Upsample(ch, conv_resample,
                                                                                             dims=dims,
                                                                                             out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def make_dense_hint_block(self, in_channels, out_channels, target_ratio, source_ratio):
        ops = [nn.SiLU()]

        scale_factor = tuple(
            source_ratio[i] // target_ratio[i] if source_ratio[i] > target_ratio[i] else 1
            for i in range(3)
        )
        if scale_factor != (1, 1, 1):
            ops.append(
                nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=False)
            )

        stride = tuple(
            target_ratio[i] // source_ratio[i] if target_ratio[i] > source_ratio[i] else 1
            for i in range(3)
        )

        ops.append(
            zero_module(
                conv_nd(3, in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
            )
        )

        return nn.Sequential(*ops)

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        condition = x[:, 1:, :, :, :]
        condition = self.cond_pre(condition)

        h = x.type(self.dtype)

        num = 0
        for module in self.input_blocks:
            num = num + 1
            h = module(h, emb)
            hint_id = (num - 2) // 2
            if num % 2 == 0 and hint_id < len(self.dense_hint_blocks):
                cond = self.dense_hint_blocks[hint_id](condition)
                h = h + cond
            hs.append(h)
        h = self.middle_block(h, emb)

        # for i in range(len(hs)):
        #     print(hs[i].shape)

        for module in self.output_blocks:
            # print('------')
            # print("h shape: ", h.shape)
            # print("hs[-1] shape: ", hs[-1].shape)
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


def create_neurodiff(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult="",
        learn_sigma=False,
        class_cond=False,
        use_checkpoint=False,
        attention_resolutions="8,16,32",
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        dropout=0,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        in_channels=1,
        out_channels=1,
        ani_ratio=1,
        res_z=10,
        res_xy=5,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        # attention_ds.append(image_size // int(res))
        attention_ds.append(int(res))

    return NeuroDiff(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=(1 * out_channels if not learn_sigma else 2 * out_channels),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        ani_ratio=ani_ratio,
        res_z=res_z,
        res_xy=res_xy,
    )
