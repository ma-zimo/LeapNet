import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConvWithShortcut(nn.Module):
    """
    Depthwise Separable Convolution with optional shortcut for block dropping.
    """
    def __init__(self, in_channels, out_channels, stride):
        super(DepthwiseSeparableConvWithShortcut, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                   stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.use_shortcut = (stride != 1) or (in_channels != out_channels)
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    '''def forward(self, x, drop):
        """
        drop: 1 to skip this block (use shortcut), 0 to execute block.
        """
        if drop.item() == 0:
            if self.use_shortcut:
                return self.shortcut(x)
            else:
                return x

        out = F.relu(self.bn1(self.depthwise(x)))
        out = F.relu(self.bn2(self.pointwise(out)))

        if self.use_shortcut:
            return out + self.shortcut(x)
        else:
            return out + x'''

    def forward(self, x, drop):
        """
        x: Tensor of shape [B, C, H, W]
        drop: Tensor of shape [B], values are 0 (skip) or 1 (execute)
        """

        if self.use_shortcut:
            shortcut_out = self.shortcut(x)
        else:
            shortcut_out = x

        if drop.item() ==1:
            B = x.size(0)
            drop = drop.view(B, 1, 1, 1)  # shape [B,1,1,1] for broadcasting

            out = F.relu(self.bn1(self.depthwise(x)))
            out = F.relu(self.bn2(self.pointwise(out)))
            output = out + shortcut_out
        else:
            output = shortcut_out
        return output


class MobileNetWithShortcut(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetWithShortcut, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList([
            DepthwiseSeparableConvWithShortcut(32, 64, 1),
            DepthwiseSeparableConvWithShortcut(64, 128, 2),
            DepthwiseSeparableConvWithShortcut(128, 128, 1),
            DepthwiseSeparableConvWithShortcut(128, 256, 2),
            DepthwiseSeparableConvWithShortcut(256, 256, 1),
            DepthwiseSeparableConvWithShortcut(256, 512, 2),
            DepthwiseSeparableConvWithShortcut(512, 512, 1),
            DepthwiseSeparableConvWithShortcut(512, 512, 1),
            DepthwiseSeparableConvWithShortcut(512, 512, 1),
            DepthwiseSeparableConvWithShortcut(512, 512, 1),
            DepthwiseSeparableConvWithShortcut(512, 512, 1),
            DepthwiseSeparableConvWithShortcut(512, 1024, 2),
            DepthwiseSeparableConvWithShortcut(1024, 1024, 1),
        ])

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        """
        x: tuple of (input_tensor, policy)
        policy: list/tensor of 0 (use block) or 1 (skip block)
        """
        x, policy = x
        # print("policy size is ", policy.size())
        x = self.stem(x)

        for i, block in enumerate(self.blocks):
            drop = policy[:, i]  # 取第 i 个 block 的所有样本的 drop 值（shape: [B]）
            # drop = int(drop.item()) if isinstance(drop, torch.Tensor) else int(drop)
            x = block(x, drop=drop)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def skip_mobilenet_inf(num_classes=10):
    return MobileNetWithShortcut(num_classes=num_classes)
