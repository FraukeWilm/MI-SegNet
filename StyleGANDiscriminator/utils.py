import torch
import torch.nn.functional as F
from torch import nn
import math



class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    dims = [1, -1] + [1] * (input.dim() - 2)
    bias = bias.view(*dims)
    return F.leaky_relu(input + bias, negative_slope) * scale


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    out = upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])

    return out

class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


def upfirdn2d_native(
        input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    bs, ch, in_h, in_w = input.shape
    minor = 1
    kernel_h, kernel_w = kernel.shape

    # assert kernel_h == 1 and kernel_w == 1

    # print("original shape ", input.shape, up_x, down_x, pad_x0, pad_x1)

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    if up_x > 1 or up_y > 1:
        out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])

    # print("after padding ", out.shape)
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    # print("after reshaping ", out.shape)

    if pad_x0 > 0 or pad_x1 > 0 or pad_y0 > 0 or pad_y1 > 0:
        out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])

    # print("after second padding ", out.shape)
    out = out[
          :,
          max(-pad_y0, 0): out.shape[1] - max(-pad_y1, 0),
          max(-pad_x0, 0): out.shape[2] - max(-pad_x1, 0),
          :,
          ]

    # print("after trimming ", out.shape)

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )

    # print("after reshaping", out.shape)
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)

    # print("after conv ", out.shape)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )

    out = out.permute(0, 2, 3, 1)

    # print("after permuting ", out.shape)

    out = out[:, ::down_y, ::down_x, :]

    out = out.view(bs, ch, out.size(1), out.size(2))

    # print("final shape ", out.shape)

    return out

