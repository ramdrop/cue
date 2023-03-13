# -*- coding: future_fstrings -*-
import sys
sys.path.append('..')
import torch
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from models import common

class ResUNet2(ME.MinkowskiNetwork):
    NORM_TYPE = None
    BLOCK_NORM_TYPE = 'BN'
    CHANNELS = [None, 32, 64, 128, 256]
    TR_CHANNELS = [None, 32, 64, 64, 128]

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling initialize_coords
    def __init__(self, in_channels=3, out_channels=32, bn_momentum=0.1, normalize_feature=None, conv1_kernel_size=None, D=3):
        ME.MinkowskiNetwork.__init__(self, D)
        NORM_TYPE = self.NORM_TYPE                         #  BN
        BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE             # BN
        CHANNELS = self.CHANNELS
        TR_CHANNELS = self.TR_CHANNELS
        self.normalize_feature = normalize_feature         # True

        # 1 > 32
        self.conv1 = ME.MinkowskiConvolution(in_channels=in_channels, out_channels=CHANNELS[1], kernel_size=conv1_kernel_size, stride=1, dilation=1, bias=False, dimension=D)
        self.norm1 = common.get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)
        self.block1 = common.get_block(BLOCK_NORM_TYPE, CHANNELS[1], CHANNELS[1], bn_momentum=bn_momentum, D=D)

        # 32 > 64
        self.conv2 = ME.MinkowskiConvolution(in_channels=CHANNELS[1], out_channels=CHANNELS[2], kernel_size=3, stride=2, dilation=1, bias=False, dimension=D)
        self.norm2 = common.get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, D=D)
        self.block2 = common.get_block(BLOCK_NORM_TYPE, CHANNELS[2], CHANNELS[2], bn_momentum=bn_momentum, D=D)

        # 64 > 128
        self.conv3 = ME.MinkowskiConvolution(in_channels=CHANNELS[2], out_channels=CHANNELS[3], kernel_size=3, stride=2, dilation=1, bias=False, dimension=D)
        self.norm3 = common.get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, D=D)
        self.block3 = common.get_block(BLOCK_NORM_TYPE, CHANNELS[3], CHANNELS[3], bn_momentum=bn_momentum, D=D)

        # 128 > 256
        self.conv4 = ME.MinkowskiConvolution(in_channels=CHANNELS[3], out_channels=CHANNELS[4], kernel_size=3, stride=2, dilation=1, bias=False, dimension=D)
        self.norm4 = common.get_norm(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum, D=D)
        self.block4 = common.get_block(BLOCK_NORM_TYPE, CHANNELS[4], CHANNELS[4], bn_momentum=bn_momentum, D=D)

        # 256 > 128
        self.conv4_tr = ME.MinkowskiConvolutionTranspose(in_channels=CHANNELS[4], out_channels=TR_CHANNELS[4], kernel_size=3, stride=2, dilation=1, bias=False, dimension=D)
        self.norm4_tr = common.get_norm(NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)
        self.block4_tr = common.get_block(BLOCK_NORM_TYPE, TR_CHANNELS[4], TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

        # 128 + 128 > 64
        self.conv3_tr = ME.MinkowskiConvolutionTranspose(in_channels=CHANNELS[3] + TR_CHANNELS[4], out_channels=TR_CHANNELS[3], kernel_size=3, stride=2, dilation=1, bias=False, dimension=D)
        self.norm3_tr = common.get_norm(NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)
        self.block3_tr = common.get_block(BLOCK_NORM_TYPE, TR_CHANNELS[3], TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

        # 64 + 64 > 64
        self.conv2_tr = ME.MinkowskiConvolutionTranspose(in_channels=CHANNELS[2] + TR_CHANNELS[3], out_channels=TR_CHANNELS[2], kernel_size=3, stride=2, dilation=1, bias=False, dimension=D)
        self.norm2_tr = common.get_norm(NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)
        self.block2_tr = common.get_block(BLOCK_NORM_TYPE, TR_CHANNELS[2], TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

        # 32 + 64 > 32
        self.conv1_tr = ME.MinkowskiConvolution(in_channels=CHANNELS[1] + TR_CHANNELS[2], out_channels=TR_CHANNELS[1], kernel_size=1, stride=1, dilation=1, bias=False, dimension=D)
        # self.block1_tr = BasicBlockBN(TR_CHANNELS[1], TR_CHANNELS[1], bn_momentum=bn_momentum, D=D)

        # 32 > 32
        self.final = ME.MinkowskiConvolution(in_channels=TR_CHANNELS[1], out_channels=out_channels, kernel_size=1, stride=1, dilation=1, bias=True, dimension=D)

    def forward(self, x):              # ([N, 1])
        out_s1 = self.conv1(x)         # ([N, 32])
        out_s1 = self.norm1(out_s1)    # ([N, 32])
        out_s1 = self.block1(out_s1)   # ([N, 32])
        out = MEF.relu(out_s1)

        out_s2 = self.conv2(out)
        out_s2 = self.norm2(out_s2)
        out_s2 = self.block2(out_s2)   # ([N1, 64])
        out = MEF.relu(out_s2)

        out_s4 = self.conv3(out)
        out_s4 = self.norm3(out_s4)
        out_s4 = self.block3(out_s4)   # ([N2, 128])
        out = MEF.relu(out_s4)

        out_s8 = self.conv4(out)
        out_s8 = self.norm4(out_s8)
        out_s8 = self.block4(out_s8)   # ([N3, 256])
        out = MEF.relu(out_s8)         # global descriptor
        # out_middle = out

        out = self.conv4_tr(out)
        out = self.norm4_tr(out)
        out = self.block4_tr(out)      # ([N2, 128])
        out_s4_tr = MEF.relu(out)

        out = ME.cat(out_s4_tr, out_s4)                    # ([N2, 128]) + ([N2, 128])

        out = self.conv3_tr(out)
        out = self.norm3_tr(out)
        out = self.block3_tr(out)      # ([N1, 64])
        out_s2_tr = MEF.relu(out)

        out = ME.cat(out_s2_tr, out_s2)                    # ([N1, 64]) + ([N1, 64])

        out = self.conv2_tr(out)
        out = self.norm2_tr(out)
        out = self.block2_tr(out)      # ([N, 64])
        out_s1_tr = MEF.relu(out)

        out = ME.cat(out_s1_tr, out_s1)                    # ([N, 96]) = ([N, 64]) + ([N, 32])
        out = self.conv1_tr(out)
        out = MEF.relu(out)                                # ([N, 64])
        out = self.final(out)                              # ([N, 32])

        if self.normalize_feature:
            return ME.SparseTensor(out.F / torch.norm(out.F, p=2, dim=1, keepdim=True), coordinate_map_key=out.coordinate_map_key, coordinate_manager=out.coordinate_manager)
            # ME.SparseTensor(out_middle.F / torch.norm(out_middle.F, p=2, dim=1, keepdim=True), coordinate_map_key=out_middle.coordinate_map_key, coordinate_manager=out_middle.coordinate_manager)
        else:
            return out


class ResUNetBN2(ResUNet2):
    NORM_TYPE = 'BN'


class ResUNetBN2B(ResUNet2):
    NORM_TYPE = 'BN'
    CHANNELS = [None, 32, 64, 128, 256]
    TR_CHANNELS = [None, 64, 64, 64, 64]


class ResUNetBN2C(ResUNet2):
    NORM_TYPE = 'BN'
    CHANNELS = [None, 32, 64, 128, 256]
    TR_CHANNELS = [None, 64, 64, 64, 128]


class ResUNetBN2D(ResUNet2):
    NORM_TYPE = 'BN'
    CHANNELS = [None, 32, 64, 128, 256]
    TR_CHANNELS = [None, 64, 64, 128, 128]


class ResUNetBN2E(ResUNet2):
    NORM_TYPE = 'BN'
    CHANNELS = [None, 128, 128, 128, 256]
    TR_CHANNELS = [None, 64, 128, 128, 128]


class ResUNetIN2(ResUNet2):
    NORM_TYPE = 'BN'
    BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2B(ResUNetBN2B):
    NORM_TYPE = 'BN'
    BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2C(ResUNetBN2C):
    NORM_TYPE = 'BN'
    BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2D(ResUNetBN2D):
    NORM_TYPE = 'BN'
    BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2E(ResUNetBN2E):
    NORM_TYPE = 'BN'
    BLOCK_NORM_TYPE = 'IN'

