import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

class CELoss(torch.nn.Module):
    def __init__(self, reduction='mean', ignore_index=-1):
        super(CELoss, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        loss = F.cross_entropy(output, target.long(), reduction=self.reduction, ignore_index=self.ignore_index)
        return loss

class DiceLoss(torch.nn.Module):
    def __init__(self, eps=0.0001, ignore_index=-1):
        super(DiceLoss, self).__init__()
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, output, target):
        output = torch.softmax(output, dim=1)
        encoded_target = output.detach() * 0
        if self.ignore_index is not None:
            mask = target == self.ignore_index
            target[mask] = 0
            encoded_target.scatter_(1, target.long().unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.long().unsqueeze(1), 1)

        intersection = output * encoded_target
        numerator = intersection.sum(0)
        denominator = output + encoded_target

        if self.ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0)

        loss = 1 - ((2 * numerator.sum() + self.eps) / (denominator.sum() + self.eps))
        return loss

class L1Loss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(L1Loss, self).__init__()
        self.reduction = reduction

    def forward(self, recon_result, input):
        loss = F.l1_loss(recon_result, input, reduction = self.reduction)
        return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

# https://github.com/Po-Hsun-Su/pytorch-ssim
class SSIMLoss(torch.nn.Module):
    def __init__(self, device, window_size = 11, size_average = True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel).to(device)
        self.device = device

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(self.device)
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

class DiscLoss(torch.nn.Module):
    def __int__(self):
        super(DiscLoss, self).__init__()

    def forward(self, pred, should_be_classified_as_real):
            bs = pred.size(0)
            if should_be_classified_as_real:
                return F.softplus(-pred).view(bs, -1).mean()
            else:
                return F.softplus(pred).view(bs, -1).mean()