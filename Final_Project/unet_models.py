import torch
from torch import nn
from torch.nn import functional as F

import utils


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


def concat(xs):
    return torch.cat(xs, 1)


class Conv3BN(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """Residual block with SE attention"""
    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1,
        downsample=None, use_se=True
    ):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if use_se:
            self.se = SEBlock(planes)
        else:
            self.se = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# 先讓 conv3x3 能接受 stride 參數
def conv3x3_2(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class ResidualSEModule(nn.Module):
    """
    兩層 3×3 Conv + SE + Residual
    與 UNetModule 介面相同 (in_ch, out_ch)，方便直接替換
    """
    def __init__(self, in_ch: int, out_ch: int, reduction: int = 16):
        super().__init__()

        self.conv1 = conv3x3_2(in_ch, out_ch, stride=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3_2(out_ch, out_ch, stride=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # SE
        self.se = SEBlock(out_ch, reduction=reduction)

        # 如果 in_ch != out_ch，就用 1×1 Conv 做 downsample
        self.downsample = None
        if in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)            # channel attention

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity                # residual add
        out = self.relu(out)
        return out


class UNetModule(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.l1 = Conv3BN(in_, out)
        self.l2 = Conv3BN(out, out)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


class UNet(nn.Module):
    module = ResidualSEModule

    def __init__(self,
                 input_channels: int = 3,
                 filters_base: int = 32,
                 filter_factors=(1, 2, 4, 8, 16)):
        super().__init__()
        filter_sizes = [filters_base * s for s in filter_factors]
        self.down, self.up = [], []
        for i, nf in enumerate(filter_sizes):
            low_nf = input_channels if i == 0 else filter_sizes[i - 1]
            self.down.append(self.module(low_nf, nf))
            setattr(self, 'down_{}'.format(i), self.down[-1])
            if i != 0:
                self.up.append(self.module(low_nf + nf, low_nf))
                setattr(self, 'conv_up_{}'.format(i), self.up[-1])
        bottom_s = 4
        pool = nn.MaxPool2d(2, 2)
        pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
        upsample = nn.UpsamplingNearest2d(scale_factor=2)
        upsample_bottom = nn.UpsamplingNearest2d(scale_factor=bottom_s)
        self.downsamplers = [None] + [pool] * (len(self.down) - 1)
        self.downsamplers[-1] = pool_bottom
        self.upsamplers = [upsample] * len(self.up)
        self.upsamplers[-1] = upsample_bottom
        self.conv_final = nn.Conv2d(filter_sizes[0], utils.N_CLASSES + 1, 1)

    def forward(self, x):
        xs = []
        for downsample, down in zip(self.downsamplers, self.down):
            x_in = x if downsample is None else downsample(xs[-1])
            x_out = down(x_in)
            xs.append(x_out)

        x_out = xs[-1]
        for x_skip, upsample, up in reversed(
                list(zip(xs[:-1], self.upsamplers, self.up))):
            x_out = upsample(x_out)
            x_out = up(concat([x_out, x_skip]))

        x_out = self.conv_final(x_out)
        return F.log_softmax(x_out)


class UNetWithHead(nn.Module):
    filters_base = 32
    unet_filters_base = 128
    unet_filter_factors = [1, 2, 4]

    def __init__(self):
        super().__init__()
        b = self.filters_base
        self.head = nn.Sequential(
            Conv3BN(3, b),
            Conv3BN(b, b),
            nn.MaxPool2d(2, 2),
            Conv3BN(b, b * 2),
            Conv3BN(b * 2, b * 2),
            nn.MaxPool2d(2, 2),
        )
        self.unet = UNet(
            input_channels=64,
            filters_base=self.unet_filters_base,
            filter_factors=self.unet_filter_factors,
        )

    def forward(self, x):
        x = self.head(x)
        return self.unet(x)


class Loss:
    def __init__(self, dice_weight=1.0, bg_weight=1.0):
        if bg_weight != 1.0:
            nll_weight = torch.ones(utils.N_CLASSES + 1)
            nll_weight[utils.N_CLASSES] = bg_weight
            nll_weight = nll_weight.cuda()
        else:
            nll_weight = None
        self.nll_loss = nn.NLLLoss2d(weight=nll_weight)
        self.dice_weight = dice_weight

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)
        if self.dice_weight:
            cls_weight = self.dice_weight / utils.N_CLASSES
            eps = 1e-5
            for cls in range(utils.N_CLASSES):
                dice_target = (targets == cls).float()
                dice_output = outputs[:, cls].exp()
                intersection = (dice_output * dice_target).sum()
                # union without intersection
                uwi = dice_output.sum() + dice_target.sum() + eps
                loss += (1 - intersection / uwi) * cls_weight
            loss /= (1 + self.dice_weight)
        return loss
