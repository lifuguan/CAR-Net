# -*- coding: utf-8 -*-

#
# Title: Stand-Alone Self-Attention in Vision Models
# Arxiv: https://arxiv.org/abs/1906.05909
# Source: https://github.com/Podidiving/unet_plus_attention
# Date: 2021-02-04
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchsummary import summary


class BlockBaseClass(nn.Module):
    """
    Class to inherit from, when creating new blocks
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 1,
            groups: int = 1,
            bias: bool = False
    ):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._groups = groups
        self._bias = bias

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def kernel_size(self):
        return self._kernel_size

    @property
    def stride(self):
        return self._stride

    @property
    def padding(self):
        return self._padding

    @property
    def groups(self):
        return self._groups

    def forward(self, x):
        raise NotImplementedError



def check_dims(
        out_channels: int,
        groups: int,
        in_channels: int
):
    """
    Check, that dimensions in attention block is Ok
    :param out_channels:
    :param groups:
    :param in_channels:
    :return:
    """
    assert not out_channels % 2, \
        f"number of out channels ({out_channels})" \
        " must be divisible by 2"

    assert not out_channels % groups, \
        f"number of out channels ({out_channels})" \
        f" must be divisible by number of groups ({groups})"

    assert not in_channels % groups, \
        f"number of in channels ({in_channels})" \
        f" must be divisible by number of groups ({groups})"

    # this assertion for embeddings
    assert not (out_channels // groups) % 2, \
        f"number of out channels ({out_channels})" \
        f" divided by number of groups ({groups})" \
        f" must be divisible by 2"



class SelfAttentionBlock(BlockBaseClass):
    """
    Main class, that implements logic of attention block of the paper
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 1,
            groups: int = 1,
            bias: bool = False
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            bias
        )

        check_dims(self._out_channels, self._groups, self._in_channels)

        self.__h_embedding: nn.Parameter = nn.Parameter(
            torch.randn(
                self._groups,
                (self._out_channels // self._groups) // 2,
                1, 1,
                self._kernel_size, 1
            ),
            requires_grad=True
        )
        self.__w_embedding: nn.Parameter = nn.Parameter(
            torch.randn(
                self._groups,
                (self._out_channels // self._groups) // 2,
                1, 1,
                1, self._kernel_size
            ),
            requires_grad=True
        )

        if self._bias:
            self.__bias = nn.Parameter(
                torch.randn(1, self._out_channels, 1, 1),
                requires_grad=True
            )

        self.__key:  nn.Conv2d = self.__create_conv()
        self.__query: nn.Conv2d = self.__create_conv()
        self.__value: nn.Conv2d = self.__create_conv()

        self.reset_params()

    def __create_conv(self):
        return nn.Conv2d(
            self._in_channels,
            self._out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            groups=self._groups
        )

    def reset_params(self):
        init.kaiming_normal_(self.__key.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.__query.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.__value.weight, mode='fan_out', nonlinearity='relu')

        if self._bias:
            bound = 1 / torch.sqrt(self._out_channels).item()
            init.uniform_(self.__bias, -bound, bound)

        init.normal_(self.__w_embedding, 0, 1)
        init.normal_(self.__h_embedding, 0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape

        x_padded: torch.Tensor = F.pad(
            x, [
                self._padding,
                self._padding,
                self._padding,
                self._padding
            ]
        )

        # [B, C_in, H, W] -> [B, C_out, H, W]
        query_out: torch.Tensor = self.__query(x)
        # [B, C_in, H, W] -> [B, C_out, H + 2*padding, W + 2*padding]
        value_out: torch.Tensor = self.__value(x_padded)
        key_out: torch.Tensor = self.__key(x_padded)

        # (H + 2*padding - kernel_size) // stride + 1 == H
        # (W + 2*padding - kernel_size) // stride + 1 == W
        # [B, C_out, H + 2*padding, W + 2*padding] ->
        # [B, C_out, H, W, kernel_size, kernel_size]
        value_out = value_out.unfold(2, self._kernel_size, self._stride)
        value_out = value_out.unfold(3, self._kernel_size, self._stride)
        key_out = key_out.unfold(2, self._kernel_size, self._stride)
        key_out = key_out.unfold(3, self._kernel_size, self._stride)

        # [B, C_out, H, W, kernel_size, kernel_size] ->
        # -> [B, groups, C_out // groups, H, W, kernel_size**2]
        value_out = value_out.contiguous().view(
            batch,
            self._groups,
            self._out_channels // self._groups,
            height,
            width,
            self._kernel_size ** 2
        )
        key_out = key_out.contiguous().view(
            batch,
            self._groups,
            self._out_channels // self._groups,
            height,
            width,
            self._kernel_size ** 2
        )

        # [B, C_out, H, W] ->
        # [B, groups, C_out // groups, H, W, 1]
        query_out = query_out.contiguous().view(
            batch,
            self._groups,
            self._out_channels // self._groups,
            height,
            width,
            1
        )

        # [groups, C_out // 2 // groups, 1, 1, kernel_size, 1] +
        # [groups, C_out // 2 // groups, 1, 1, 1, kernel_size] ->
        # [groups, C_out // groups, 1, 1, kernel_size, kernel_size]
        hw_emb: torch.Tensor = torch.cat(
            [
                self.__h_embedding.repeat(1, 1, 1, 1, 1, self._kernel_size),
                self.__w_embedding.repeat(1, 1, 1, 1, self._kernel_size, 1)
            ],
            dim=1
        )
        # [groups, C_out // groups, 1, 1, kernel_size, kernel_size] ->
        # [groups, C_out // groups, 1, 1, kernel_size**2]
        hw_emb = hw_emb.contiguous().view(
            self._groups,
            self._out_channels // self._groups,
            1, 1,
            self._kernel_size**2
        )

        plus = key_out + hw_emb
        # -> [B, groups, C_out // groups, H, W, kernel_size**2]
        out: torch.Tensor = query_out * plus
        out = F.softmax(out, dim=-1)

        # -> [B, groups, C_out // groups, H, W]
        out = torch.einsum("bgchwk,bgchwk -> bgchw", [out, value_out])

        # -> [B, C_out, H, W]
        out = out.view(batch, self._out_channels, height, width)
        if self._bias:
            out += self.__bias

        return out



def vanila_block_build(in_: int, out: int) -> nn.ModuleList:
    return nn.ModuleList([
        nn.Conv2d(in_, out, 3, padding=1),
        nn.BatchNorm2d(out),
        nn.ReLU(),
        nn.Conv2d(out, out, 3, padding=1),
        nn.BatchNorm2d(out),
        nn.ReLU()
    ])


def attention_block_build(in_: int, out: int) -> nn.ModuleList:
    return nn.ModuleList([
        nn.Conv2d(in_, in_ // 4, kernel_size=1),
        nn.BatchNorm2d(in_ // 4),
        nn.ReLU(),
        SelfAttentionBlock(in_ // 4, out // 4, kernel_size=5, stride=1, padding=2, groups=4),
        nn.BatchNorm2d(out // 4),
        nn.ReLU(),
        nn.Conv2d(out // 4, out, kernel_size=1),
        nn.BatchNorm2d(out),
        nn.ReLU()
    ])


class AttentionWithSkipConnection(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_, in_ // 4, kernel_size=1),
            nn.BatchNorm2d(in_ // 4),
            nn.ReLU(),
            SelfAttentionBlock(in_ // 4, out // 4, kernel_size=5, stride=1, padding=2, groups=4),
            nn.BatchNorm2d(out // 4),
            nn.ReLU(),
            nn.Conv2d(out // 4, out, kernel_size=1),
            nn.BatchNorm2d(out),
            nn.ReLU()
        )
        if in_ != out:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_, out, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out)
            )
        else:
            self.downsample = None

    def forward(self, x):
        out = self.seq(x)
        if self.downsample is not None:
            out += self.downsample(x)
        else:
            out += x
        return out


class DepthWiseSeparableConv(nn.Module):
    def __init__(self, in_, out, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_, in_, kernel_size=kernel_size,
                stride=stride, padding=padding, groups=in_
            ),
            nn.ReLU(),
            nn.BatchNorm2d(in_),
            nn.Conv2d(in_, out, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)


def attention_with_skip_connection_block_build(in_: int, out: int) -> nn.ModuleList:
    return nn.ModuleList([
        AttentionWithSkipConnection(in_, out)
    ])


def depthwise_separable_block_build(in_: int, out: int) -> nn.ModuleList:
    return nn.ModuleList([
        DepthWiseSeparableConv(in_, out, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out),
        DepthWiseSeparableConv(out, out, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out),
    ])


def depthwise_separable_block_light_build(in_: int, out: int) -> nn.ModuleList:
    return nn.ModuleList([
        DepthWiseSeparableConv(in_, out, padding=1),
    ])



class Self_Attention_UNet(nn.Module):
    def __init__(self, module_builders: dict, in_dim=3, out_dim=1):
        super().__init__()

        self.encoder1 = nn.Sequential(
            *module_builders['enc1'](in_dim, 64),
        )
        self.encoder2 = nn.Sequential(
            *module_builders['enc2'](64, 128),
        )
        self.encoder3 = nn.Sequential(
            *module_builders['enc3'](128, 256),
        )
        self.encoder4 = nn.Sequential(
            *module_builders['enc4'](256, 512),
        )
        self.encoder5 = nn.Sequential(
            *module_builders['enc5'](512, 1024),
        )

        self.transpose: nn.ModuleList = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, 2, stride=2),
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
        ])

        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    *module_builders['dec1'](1024, 512),
                ),
                nn.Sequential(
                    *module_builders['dec2'](512, 256),
                ),
                nn.Sequential(
                    *module_builders['dec3'](256, 128),
                ),
                nn.Sequential(
                    *module_builders['dec4'](128, 64),
                )
            ]
        )

        self.clf = nn.Conv2d(64, out_dim, 1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))
        enc5 = self.encoder5(F.max_pool2d(enc4, 2))
        tr1 = self.transpose[0](enc5)
        dec1 = self.decoders[0](torch.cat([enc4, tr1], 1))

        tr2 = self.transpose[1](dec1)
        dec2 = self.decoders[1](torch.cat([enc3, tr2], 1))

        tr3 = self.transpose[2](dec2)
        dec3 = self.decoders[2](torch.cat([enc2, tr3], 1))

        tr4 = self.transpose[3](dec3)
        dec4 = self.decoders[3](torch.cat([enc1, tr4], 1))
        return self.clf(dec4)


def get_vanila_unet_model(in_dim: int, out_dim: int) -> nn.Module:
    module_builders = {
        'enc1': vanila_block_build,
        'enc2': vanila_block_build,
        'enc3': vanila_block_build,
        'enc4': vanila_block_build,
        'enc5': vanila_block_build,
        'dec1': vanila_block_build,
        'dec2': vanila_block_build,
        'dec3': vanila_block_build,
        'dec4': vanila_block_build,
    }
    return Self_Attention_UNet(module_builders, in_dim, out_dim)


def get_unet_attention_decoder(in_dim: int, out_dim: int) -> nn.Module:
    module_builders = {
        'enc1': vanila_block_build,
        'enc2': vanila_block_build,
        'enc3': vanila_block_build,
        'enc4': vanila_block_build,
        'enc5': vanila_block_build,
        'dec1': attention_block_build,
        'dec2': attention_block_build,
        'dec3': attention_block_build,
        'dec4': attention_block_build,
    }
    return Self_Attention_UNet(module_builders, in_dim, out_dim)


def get_unet_attention_with_skip_connections_decoder(
        in_dim: int,
        out_dim: int
) -> nn.Module:
    module_builders = {
        'enc1': vanila_block_build,
        'enc2': vanila_block_build,
        'enc3': vanila_block_build,
        'enc4': vanila_block_build,
        'enc5': vanila_block_build,
        'dec1': attention_with_skip_connection_block_build,
        'dec2': attention_with_skip_connection_block_build,
        'dec3': attention_with_skip_connection_block_build,
        'dec4': attention_with_skip_connection_block_build,
    }
    return Self_Attention_UNet(module_builders, in_dim, out_dim)


def get_unet_depthwise_encoder_attention_with_skip_connections_decoder(
        in_dim: int,
        out_dim: int
) -> nn.Module:
    module_builders = {
        'enc1': depthwise_separable_block_build,
        'enc2': depthwise_separable_block_build,
        'enc3': depthwise_separable_block_build,
        'enc4': depthwise_separable_block_build,
        'enc5': depthwise_separable_block_build,
        'dec1': attention_with_skip_connection_block_build,
        'dec2': attention_with_skip_connection_block_build,
        'dec3': attention_with_skip_connection_block_build,
        'dec4': attention_with_skip_connection_block_build,
    }
    return Self_Attention_UNet(module_builders, in_dim, out_dim)


def get_unet_depthwise_light_encoder_attention_with_skip_connections_decoder(
        in_dim: int,
        out_dim: int
) -> nn.Module:
    module_builders = {
        'enc1': depthwise_separable_block_light_build,
        'enc2': depthwise_separable_block_light_build,
        'enc3': depthwise_separable_block_light_build,
        'enc4': depthwise_separable_block_light_build,
        'enc5': depthwise_separable_block_light_build,
        'dec1': attention_with_skip_connection_block_build,
        'dec2': attention_with_skip_connection_block_build,
        'dec3': attention_with_skip_connection_block_build,
        'dec4': attention_with_skip_connection_block_build,
    }
    return Self_Attention_UNet(module_builders, in_dim, out_dim)



"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_unet_depthwise_light_encoder_attention_with_skip_connections_decoder(3,1).to(device)
    summary(model,(3,128,128))       # Succeed!