import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super(SqueezeExcitation, self).__init__()
        reduced_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, 1),
            Swish(),
            nn.Conv2d(reduced_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Swish()
            ]
        layers += [
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Swish(),
            SqueezeExcitation(hidden_dim, se_ratio)  # 正确插入位置
        ]
        layers += [
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x, policy):
        policy = policy.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        # print("spec: x's size {}, policy's size {}".format(x.size(), policy.size()))
        out = self.block(x)
        if self.use_residual:
            out = out * policy + x
        else:
            out = out
        return out


class CustomLayer(nn.Module):
    def __init__(self, blocks):
        super(CustomLayer, self).__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, policy):
        start_idx = 0
        for block in self.blocks:
            x = block(x, policy[:, start_idx])
            # x = block(x, policy[start_idx])  # when batchsize =1
            start_idx += 1
        return x


class EfficientNet(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, depth_mult=1.0):
        super(EfficientNet, self).__init__()
        base_channels = 32
        last_channels = 1280
        settings = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 40, 2, 2],
            [6, 80, 3, 2],
            [6, 112, 3, 1],
            [6, 192, 4, 2],
            [6, 320, 1, 1],
        ]

        def round_filters(filters):
            return int(filters * width_mult)

        def round_repeats(repeats):
            return int(math.ceil(repeats * depth_mult))

        out_channels = round_filters(base_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            Swish()
        )
        in_channels = out_channels
        layers = []
        for t, c, n, s in settings:
            out_channels = round_filters(c)
            blocks = []
            for i in range(round_repeats(n)):
                stride = s if i == 0 else 1
                blocks.append(MBConvBlock(in_channels, out_channels, t, stride))
                in_channels = out_channels
            layers.append(CustomLayer(blocks))

        # self.blocks = nn.Sequential(*layers)
        self.blocks = nn.ModuleList(layers)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, round_filters(last_channels), 1, bias=False),
            nn.BatchNorm2d(round_filters(last_channels)),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.4),  #### dropout
            nn.Linear(round_filters(last_channels), num_classes)
        )

    def forward(self, x):
        x, policy = x
        # print("x's size {} and policy's size {}".format(x.size(), policy.size()))
        x = self.stem(x)
        # x = self.blocks(x)
        start = 0
        for layer in self.blocks:
            num_blocks = len(layer.blocks)
            # print('num_block is',num_blocks)
            x = layer(x, policy[:, start:start + num_blocks])
            start += num_blocks
        x = self.head(x)
        # print('total num_block is',start)
        return x


def skip_efficientnet(name='efficientnet_b0', num_classes=1000):
    width_mult, depth_mult = {
        'efficientnet_b0': (1.0, 1.0),
        'efficientnet_b1': (1.0, 1.1),
        'efficientnet_b2': (1.1, 1.2),
        'efficientnet_b3': (1.2, 1.4),
        'efficientnet_b4': (1.4, 1.8),
        'efficientnet_b5': (1.6, 2.2),
        'efficientnet_b6': (1.8, 2.6),
        'efficientnet_b7': (2.0, 3.1),
    }.get(name, (1.0, 1.0))

    return EfficientNet(num_classes=num_classes, width_mult=width_mult, depth_mult=depth_mult)
