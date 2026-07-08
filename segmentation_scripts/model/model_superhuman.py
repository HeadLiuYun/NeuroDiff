# deployed model without much flexibility
# useful for stand-alone test, model translation, quantization
import torch.nn as nn
import torch.nn.functional as F
import torch


def init_conv(m, init_mode):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
        if init_mode == 'kaiming_normal':
            nn.init.kaiming_normal_(m.weight)
        elif init_mode == 'kaiming_uniform':
            nn.init.kaiming_uniform_(m.weight)
        elif init_mode == 'xavier_normal':
            nn.init.xavier_normal_(m.weight)
        elif init_mode == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight)

        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def getConv3d(in_planes, out_planes, kernel_size, stride, padding,
              bias, pad_mode='zero', init_mode='', dilation_size=(1, 1, 1)):
    out = []
    if pad_mode == 'zero':  # 0-padding
        out = [nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, \
                         dilation=dilation_size, padding=padding, stride=stride, bias=bias)]
    elif pad_mode == 'replicate':  # replication-padding
        # need 6 values
        padding = tuple([x for x in padding for _ in range(2)][::-1])
        out = [nn.ReplicationPad3d(padding),
               nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                         stride=stride, dilation=dilation_size, bias=bias)]
    if len(out) == 0:
        raise ValueError('Unknown padding option {}'.format(mode))
    else:
        if init_mode != '':  # do conv init
            init_conv(out[-1], init_mode)
        return out


def getRelu(mode='relu'):
    if mode == 'relu':
        return nn.ReLU(inplace=True)
    elif mode == 'elu':
        return nn.ELU(inplace=True)
    elif mode[:5] == 'leaky':
        return nn.LeakyReLU(inplace=True, negative_slope=float(mode[5:]))
    raise ValueError('Unknown ReLU option {}'.format(mode))


def getBN(out_planes, dim=1, mode='sync', bn_momentum=0.1):
    if mode == 'async':
        if dim == 1:
            return nn.BatchNorm1d(out_planes, momentum=bn_momentum)
        elif dim == 2:
            return nn.BatchNorm2d(out_planes, momentum=bn_momentum)
        elif dim == 3:
            return nn.BatchNorm3d(out_planes, momentum=bn_momentum)
    elif mode == 'sync':
        if dim == 1:
            return SynchronizedBatchNorm1d(out_planes, momentum=bn_momentum)
        elif dim == 2:
            return SynchronizedBatchNorm2d(out_planes, momentum=bn_momentum)
        elif dim == 3:
            return SynchronizedBatchNorm3d(out_planes, momentum=bn_momentum)
    raise ValueError('Unknown BatchNorm option: ' + str(mode))


def upsampleBlock(in_planes, out_planes, up=(1, 2, 2), mode='bilinear',
                  kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True, init_mode=''):
    # Upsampling
    out = None
    if mode == 'bilinear':
        out = [nn.Upsample(scale_factor=up, mode='trilinear', align_corners=True),
               nn.Conv3d(in_planes, out_planes, kernel_size, stride=stride, padding=padding, bias=bias)]
    elif mode == 'nearest':
        out = [nn.Upsample(scale_factor=up, mode='nearest'),
               nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
    elif mode == 'transpose':  # dense version
        out = [nn.ConvTranspose3d(
            in_planes, out_planes, kernel_size=kernel_size,
            stride=up, bias=bias)]
    elif mode == 'transposeS':  # sparse version
        out = [nn.ConvTranspose3d(
            in_planes, in_planes, kernel_size=up,
            stride=up, bias=bias, groups=in_planes),
            nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, bias=bias)]
    if out is None:
        raise ValueError('Unknown upsampling mode {}'.format(mode))
    else:
        out = nn.Sequential(*out)
        for m in range(len(out._modules)):
            init_conv(out._modules[str(m)], init_mode)
        return out


def conv3dBlock(in_planes, out_planes, kernel_size=[(3, 3, 3)], stride=[1], padding=[0], bias=[True], pad_mode=['zero'],
                bn_mode=[''], relu_mode=[''], init_mode='kaiming_normal', bn_momentum=0.1, dilation_size=None):
    # easy to make VGG style layers
    layers = []
    if dilation_size is None:
        dilation_size = [(1, 1, 1)] * len(in_planes)
    for i in range(len(in_planes)):
        if in_planes[i] > 0:
            layers += getConv3d(in_planes[i], out_planes[i], kernel_size[i], stride[i], padding[i], bias[i],
                                pad_mode[i], init_mode, dilation_size[i])
        if bn_mode[i] != '':
            layers.append(getBN(out_planes[i], 3, bn_mode[i], bn_momentum))
        if relu_mode[i] != '':
            layers.append(getRelu(relu_mode[i]))
    return nn.Sequential(*layers)


class resBlock_pni(nn.Module):
    # https://github.com/torms3/Superhuman/blob/torch-0.4.0/code/rsunet.py#L145
    def __init__(self, in_planes, out_planes, pad_mode='zero', bn_mode='', relu_mode='', init_mode='', bn_momentum=0.1):
        super(resBlock_pni, self).__init__()
        self.block1 = conv3dBlock([in_planes], [out_planes], [(1, 3, 3)], [1], [(0, 1, 1)],
                                  [False], [pad_mode], [bn_mode], [relu_mode], init_mode, bn_momentum)
        # no relu for the second block
        self.block2 = conv3dBlock([out_planes] * 2, [out_planes] * 2, [(3, 3, 3)] * 2, [1] * 2, [(1, 1, 1)] * 2,
                                  [False] * 2, [pad_mode] * 2, [bn_mode, ''], [relu_mode, ''], init_mode, bn_momentum)
        self.block3 = getBN(out_planes, 3, bn_mode, bn_momentum)

        self.block4 = None
        if relu_mode != '':
            self.block4 = getRelu(relu_mode)

    def forward(self, x):
        residual = self.block1(x)
        out = residual + self.block2(residual)
        out = self.block3(out)
        if self.block4 is not None:
            out = self.block4(out)
        return out


class UNet_PNI(nn.Module):  # deployed PNI model
    # Superhuman Accuracy on the SNEMI3D Connectomics Challenge. Lee et al.
    # https://arxiv.org/abs/1706.00120
    def __init__(self, in_planes=1,
                 out_planes=3,
                 filters=[28, 36, 48, 64, 80],  # [28, 36, 48, 64, 80], [32, 64, 128, 256, 512]
                 upsample_mode='transposeS',  # transposeS, bilinear
                 decode_ratio=1,
                 merge_mode='cat',
                 pad_mode='zero',
                 bn_mode='async',  # async or sync
                 relu_mode='elu',
                 init_mode='kaiming_normal',
                 bn_momentum=0.001,
                 do_embed=True,
                 if_sigmoid=True,
                 show_feature=False):
        # filter_ratio: #filter_decode/#filter_encode
        super(UNet_PNI, self).__init__()
        filters2 = filters[:1] + filters
        self.merge_mode = merge_mode
        self.do_embed = do_embed
        self.depth = len(filters2) - 2
        self.if_sigmoid = if_sigmoid
        self.show_feature = show_feature

        # 2D conv for anisotropic
        self.embed_in = conv3dBlock([in_planes],
                                    [filters2[0]],
                                    [(1, 5, 5)],
                                    [1],
                                    [(0, 2, 2)],
                                    [True],
                                    [pad_mode],
                                    [''],
                                    [relu_mode],
                                    init_mode,
                                    bn_momentum)

        # downsample stream
        self.conv0 = resBlock_pni(filters2[0], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool0 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv1 = resBlock_pni(filters2[1], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = resBlock_pni(filters2[2], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = resBlock_pni(filters2[3], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.center = resBlock_pni(filters2[4], filters2[5], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        # upsample stream
        self.up0 = upsampleBlock(filters2[5], filters2[4], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat0 = conv3dBlock([0], [filters2[4]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni(filters2[4], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat0 = conv3dBlock([0], [filters2[4] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni(filters2[4] * 2, filters2[4], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.up1 = upsampleBlock(filters2[4], filters2[3], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat1 = conv3dBlock([0], [filters2[3]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni(filters2[3], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat1 = conv3dBlock([0], [filters2[3] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni(filters2[3] * 2, filters2[3], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.up2 = upsampleBlock(filters2[3], filters2[2], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat2 = conv3dBlock([0], [filters2[2]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni(filters2[2], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat2 = conv3dBlock([0], [filters2[2] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni(filters2[2] * 2, filters2[2], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.up3 = upsampleBlock(filters2[2], filters2[1], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat3 = conv3dBlock([0], [filters2[1]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni(filters2[1], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat3 = conv3dBlock([0], [filters2[1] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni(filters2[1] * 2, filters2[1], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.embed_out = conv3dBlock([int(filters2[0])],
                                     [int(filters2[0])],
                                     [(1, 5, 5)],
                                     [1],
                                     [(0, 2, 2)],
                                     [True],
                                     [pad_mode],
                                     [''],
                                     [relu_mode],
                                     init_mode,
                                     bn_momentum)

        self.out_put = conv3dBlock([int(filters2[0])], [out_planes], [(1, 1, 1)], init_mode=init_mode)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # embedding
        embed_in = self.embed_in(x)
        conv0 = self.conv0(embed_in)
        pool0 = self.pool0(conv0)
        conv1 = self.conv1(pool0)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        center = self.center(pool3)
        # print(center.shape)

        up0 = self.up0(center)
        if self.merge_mode == 'add':
            cat0 = self.cat0(up0 + conv3)
        else:
            cat0 = self.cat0(torch.cat([up0, conv3], dim=1))
        conv4 = self.conv4(cat0)

        up1 = self.up1(conv4)
        if self.merge_mode == 'add':
            cat1 = self.cat1(up1 + conv2)
        else:
            cat1 = self.cat1(torch.cat([up1, conv2], dim=1))
        conv5 = self.conv5(cat1)

        up2 = self.up2(conv5)
        if self.merge_mode == 'add':
            cat2 = self.cat2(up2 + conv1)
        else:
            cat2 = self.cat2(torch.cat([up2, conv1], dim=1))
        conv6 = self.conv6(cat2)

        up3 = self.up3(conv6)
        if self.merge_mode == 'add':
            cat3 = self.cat3(up3 + conv0)
        else:
            cat3 = self.cat3(torch.cat([up3, conv0], dim=1))
        conv7 = self.conv7(cat3)

        embed_out = self.embed_out(conv7)
        out = self.out_put(embed_out)

        if self.if_sigmoid:
            out = torch.sigmoid(out)

        if self.show_feature:
            down_features = [conv0, conv1, conv2, conv3]
            center_features = [center]
            up_features = [conv4, conv5, conv6, conv7]
            return down_features, center_features, up_features, out
        else:
            return out


if __name__ == "__main__":
    net = UNet_PNI(1, 3, show_feature=True).cuda()
    a = torch.rand(2, 1, 18, 160, 160).cuda()
    _, center, _, _ = net(a)
    center=center[0]
    print(center.shape)
    print(type(center))
