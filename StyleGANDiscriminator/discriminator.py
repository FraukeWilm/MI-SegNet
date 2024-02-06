# https://github.com/taesungp/swapping-autoencoder-pytorch/blob/main/models/networks/stylegan2_layers.py
from collections import OrderedDict
import math
import torch
from torch import nn
from torch.nn import functional as F

from StyleGANDiscriminator.utils import ScaledLeakyReLU, FusedLeakyReLU, fused_leaky_relu, upfirdn2d


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.dim() == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1, reflection_pad=False):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad
        self.reflection = reflection_pad
        if self.reflection:
            self.reflection_pad = nn.ReflectionPad2d((pad[0], pad[1], pad[0], pad[1]))
            self.pad = (0, 0)

    def forward(self, input):
        if self.reflection:
            input = self.reflection_pad(input)
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out

class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            if input.dim() > 2:
                out = F.conv2d(input, self.weight[:, :, None, None] * self.scale)
            else:
                out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            if input.dim() > 2:
                out = F.conv2d(input, self.weight[:, :, None, None] * self.scale,
                               bias=self.bias * self.lr_mul
                )
            else:
                out = F.linear(
                    input, self.weight * self.scale, bias=self.bias * self.lr_mul
                )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )

class EqualConv2d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True, lr_mul=1.0,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2) * lr_mul

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        pad=None,
        reflection_pad=False,
    ):
        layers = []

        if downsample:
            factor = 2
            if pad is None:
                pad = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (pad + 1) // 2
            pad1 = pad // 2

            layers.append(("Blur", Blur(blur_kernel, pad=(pad0, pad1), reflection_pad=reflection_pad)))

            stride = 2
            self.padding = 0
        else:
            stride = 1
            self.padding = kernel_size // 2 if pad is None else pad
            if reflection_pad:
                layers.append(("RefPad", nn.ReflectionPad2d(self.padding)))
                self.padding = 0


        layers.append(("Conv",
                       EqualConv2d(
                           in_channel,
                           out_channel,
                           kernel_size,
                           padding=self.padding,
                           stride=stride,
                           bias=bias and not activate,
                       ))
        )

        if activate:
            if bias:
                layers.append(("Act", FusedLeakyReLU(out_channel)))

            else:
                layers.append(("Act", ScaledLeakyReLU(0.2)))

        super().__init__(OrderedDict(layers))

    def forward(self, x):
        out = super().forward(x)
        return out



class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], reflection_pad=False, pad=None, downsample=True):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3, reflection_pad=reflection_pad, pad=pad)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=downsample, blur_kernel=blur_kernel, reflection_pad=reflection_pad, pad=pad)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=downsample, blur_kernel=blur_kernel, activate=False, bias=False
        )

    def forward(self, input):
        #print("before first resnet layeer, ", input.shape)
        out = self.conv1(input)
        #print("after first resnet layer, ", out.shape)
        out = self.conv2(out)
        #print("after second resnet layer, ", out.shape)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out

class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: min(512, int(512 * channel_multiplier)),
            32: min(512, int(512 * channel_multiplier)),
            64: int(256 * channel_multiplier),
            128: int(128 * channel_multiplier),
            256: int(64 * channel_multiplier),
            512: int(32 * channel_multiplier),
            1024: int(16 * channel_multiplier),
        }

        original_size = size

        size = 2 ** int(round(math.log(size, 2)))

        convs = [('0', ConvLayer(3, channels[size], 1))]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            layer_name = str(9 - i) if i <= 8 else "%dx%d" % (2 ** i, 2 ** i)
            convs.append((layer_name, ResBlock(in_channel, out_channel, blur_kernel)))

            in_channel = out_channel

        self.convs = nn.Sequential(OrderedDict(convs))

        #self.stddev_group = 4
        #self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel, channels[4], 3)

        side_length = int(4 * original_size / size)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * (side_length ** 2), channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out

    def get_features(self, input):
        return self.final_conv(self.convs(input))