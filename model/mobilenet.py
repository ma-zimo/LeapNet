import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Implements a depthwise separable convolution block"""
    def __init__(self, in_channels, out_channels, stride):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                   stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x


class MobileNet(nn.Module):
    """MobileNetV1 model"""
    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()

        def conv_dw(in_channels, out_channels, stride):
            return DepthwiseSeparableConv(in_channels, out_channels, stride)

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),

            # 5x blocks with 512 -> 512
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),

            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def mobilenet(num_classes=10, pretrained=False):
    """
    Returns MobileNetV1 model.
    Arguments:
        num_classes (int): number of output classes.
        pretrained (bool): not supported in this custom version.
    """
    if pretrained:
        raise NotImplementedError("Pretrained MobileNetV1 not supported in this implementation.")
    return MobileNet(num_classes=num_classes)
