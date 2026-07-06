import torch
import torch.nn as nn
import torch.nn.functional as F
from .wideresnet import wideresnet
from .resnet import resnet
from torch.distributions import Bernoulli

class Normalization(nn.Module):
    """
    Standardizes the input data.
    Arguments:
        mean (list): mean.
        std (float): standard deviation.
        device (str or torch.device): device to be used.
    Returns:
        (input - mean) / std
    """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        num_channels = len(mean)
        self.mean = torch.FloatTensor(mean).view(1, num_channels, 1, 1)
        self.sigma = torch.FloatTensor(std).view(1, num_channels, 1, 1)
        self.mean_cuda, self.sigma_cuda = None, None

    def forward(self, x):
        if x.is_cuda:
            if self.mean_cuda is None:
                self.mean_cuda = self.mean.cuda()
                self.sigma_cuda = self.sigma.cuda()
            out = (x - self.mean_cuda) / self.sigma_cuda
        else:
            out = (x - self.mean) / self.sigma
        return out


class BasicBlock(nn.Module):
    """
    Implements a basic block module for Resnets.
    Arguments:
        in_planes (int): number of input planes.
        out_planes (int): number of output filters.
        stride (int): stride of convolution.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x, policy):
        #print("Input tensor device:", x.device)
        #print("Input policy tensor device:", policy.device)
        out = F.relu(self.bn1(self.conv1(x)))
        #print("Input tensor device:", x.device)
        out = self.bn2(self.conv2(out))
        policy = policy.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        #print("out's size {} and policy's size {}".format(out.size(), policy.size()))
        out = out * policy + self.shortcut(x)
        out = F.relu(out)
        # if policy.item() ==1:
        #     out = F.relu(self.bn1(self.conv1(x)))
        #     out = self.bn2(self.conv2(out))
        #     out += self.shortcut(x)
        #     out = F.relu(out)
        # else:
        #     out = self.shortcut(x)
        #     out = F.relu(out)
        return out
    """
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out"""


class Bottleneck(nn.Module):
    """
    Implements a basic block module with bottleneck for Resnets.
    Arguments:
        in_planes (int): number of input planes.
        out_planes (int): number of output filters.
        stride (int): stride of convolution.
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x, policy):
        # print("out's size {} and policy's size {}".format(x.size(), policy.size()))
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        policy = policy.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        out = out * policy + self.shortcut(x)
        out = F.relu(out)
        return out


class CustomLayer(nn.Module):
    def __init__(self, blocks):
        super(CustomLayer, self).__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, policy):
        start_idx = 0
        for block in self.blocks:
            x = block(x, policy[:,start_idx])
            #x = block(x, policy[start_idx])  # when batchsize =1
            start_idx += 1
        return x
# class CustomLayer(nn.Module):
#     def __init__(self, blocks):
#         super(CustomLayer, self).__init__()
#         self.blocks = nn.ModuleList(blocks)
#
#     def forward(self, x):
#         start_idx = 0
#         for block in self.blocks:
#             x = block(x)
#             start_idx += 1
#         return x
class ResNet(nn.Module):
    """
    ResNet model
    Arguments:
        block (BasicBlock or Bottleneck): type of basic block to be used.
        num_blocks (list): number of blocks in each sub-module.
        num_classes (int): number of output classes.
        device (torch.device or str): device to work on.
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_blocks = num_blocks

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return CustomLayer(layers)

    """def forward(self, x):
        #x,policy = x
        out = F.relu(self.bn1(self.conv1(x)))
        #policy = Bernoulli(torch.full((x.size(0), sum(self.num_blocks)), 0.4)).sample().cuda()
        #policy = torch.ones(x.size(0), sum(self.num_blocks)).cuda()
        policy = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1]).cuda()
        start_idx = 0
        out = self.layer1(out,policy[start_idx:start_idx+self.num_blocks[0]])
        start_idx += self.num_blocks[0]
        out = self.layer2(out,policy[start_idx:start_idx+self.num_blocks[1]])
        start_idx += self.num_blocks[1]
        out = self.layer3(out,policy[start_idx:start_idx+self.num_blocks[2]])
        start_idx += self.num_blocks[2]
        out = self.layer4(out,policy[start_idx:start_idx+self.num_blocks[3]])
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    """
    def forward(self, x):
        x,policy = x
        # print("x's size {} and policy's size {}".format(x.size(), policy.size()))
        out = F.relu(self.bn1(self.conv1(x)))
        start_idx = 0
        out = self.layer1(out,policy[:,start_idx:start_idx+self.num_blocks[0]])
        start_idx += self.num_blocks[0]
        out = self.layer2(out,policy[:,start_idx:start_idx+self.num_blocks[1]])
        start_idx += self.num_blocks[1]
        out = self.layer3(out,policy[:,start_idx:start_idx+self.num_blocks[2]])
        start_idx += self.num_blocks[2]
        out = self.layer4(out,policy[:,start_idx:start_idx+self.num_blocks[3]])
        out = F.adaptive_avg_pool2d(out, (1, 1)) #F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    # def forward(self, x):
    #     out = F.relu(self.bn1(self.conv1(x)))
    #     out = self.layer1(out)
    #     out = self.layer2(out)
    #     out = self.layer3(out)
    #     out = self.layer4(out)
    #     out = F.avg_pool2d(out, 4)
    #     out = out.view(out.size(0), -1)
    #     out = self.linear(out)
    #     return out



def skip_resnet(name, num_classes=10, pretrained=False):
    """
    Returns suitable Resnet model from its name.
    Arguments:
        name (str): name of resnet architecture.
        num_classes (int): number of target classes.
        pretrained (bool): whether to use a pretrained model.
        device (str or torch.device): device to work on.
    Returns:
        torch.nn.Module.
    """
    if name == 'resnet18':
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    elif name == 'resnet34':
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    elif name == 'resnet50':
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    elif name == 'resnet101':
        return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
    elif name == 'resnet152':
        return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)

    raise ValueError('Only resnet18, resnet34, resnet50 and resnet101 are supported!')
    return


class PolicyNet_oldversion(nn.Module):
    def __init__(self, args, num_blocks):
        super(PolicyNet_oldversion, self).__init__()

        self.feature_extractor = wideresnet('wrn-16-1')  #wrn-16-1

        # Replace the last fully connected layer
        # Adjust the output features according to your specific number of blocks
        self.feat_dim = self.feature_extractor.fc.weight.data.shape[1]
        self.feature_extractor.fc = nn.Sequential()
        self.fc = nn.Linear(self.feat_dim, num_blocks)

    def forward(self, x):
        # Pass the input through the feature extractor
        features = self.feature_extractor(x)

        # Generate the policy for each block
        policy = torch.sigmoid(self.fc(features))
        return policy
