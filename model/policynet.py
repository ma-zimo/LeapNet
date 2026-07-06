import torch
import torch.nn as nn
import torch.nn.functional as F


class Small(nn.Module):
    def __init__(self,args):
        super(Small, self).__init__()

        self.num_classes=int(args.num_classes)
        self.in_channels=int(args.in_channels)
        self.input_size=int(args.input_size)

        self.conv1 = nn.Sequential(
                nn.Conv2d(self.in_channels, 32, 5, stride=1),
                nn.ReLU(inplace=False), # inplace=True, reduce memory
                nn.MaxPool2d(2, 2),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(32, 32, 5, stride=1),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(2, 2),
                )

        # version1: only two conv layers, then a fc for latency
        # compute the dim after two conv
        final_size = ((self.input_size - 4) // 2 - 4) // 2
        self.feat_dim_before_fc = 32 * final_size * final_size


        #version2: add a linear layer in small net
        # self.linear = nn.Linear(32 * int((self.input_size / 4 - 3) * (self.input_size / 4 - 3)), self.num_classes)

        #version4: move the linear layers in small to the latency part

    def forward(self, x):
        x = self.conv1(x)
        #print(x)
        x = self.conv2(x)

        return x.view(-1, self.feat_dim_before_fc)
        # return x.view(-1, 32*int((self.input_size/4-3)*(self.input_size/4-3)))

        # x = x.view(-1, 32 * int((self.input_size / 4 - 3) * (self.input_size / 4 - 3)))
        # # print(x)
        # x = self.linear(x)
        # return x

def small(args):
    return Small(args)

class PolicyNet(nn.Module):
    def __init__(self, args, num_blocks, device='cpu'):
        super(PolicyNet, self).__init__()

        self.feature_extractor = small(args)
        self.feat_dim = self.feature_extractor.feat_dim_before_fc
        self.fc = nn.Linear(self.feat_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_blocks)



    def forward(self, x):
        # Pass the input through the feature extractor
        features = self.feature_extractor(x)
        fc1 = self.fc(features)
        fc2 = self.fc2(torch.relu(fc1))
        fc3 = self.fc3(torch.relu(fc2))
        policy = torch.sigmoid(fc3)
        # out_temp = fc3.view(-1, 8, 2)
        # policy = torch.softmax(out_temp, dim=-1)
        return policy#[:,:,0]


class PolicyNet_L(nn.Module):
    def __init__(self, args, num_blocks, device='cpu'):
        super(PolicyNet_L, self).__init__()

        self.feature_extractor = small(args)
        self.feat_dim = self.feature_extractor.feat_dim_before_fc
        self.fc = nn.Linear(self.feat_dim + 1, 64)
        self.fc2 = nn.Linear(64 + 1, 32)
        self.fc3 = nn.Linear(32 + 1, num_blocks)


    def forward(self, x, latency):
        # Pass the input through the feature extractor
        features = self.feature_extractor(x)
        latency_values = latency.view(-1, 1)
        features_with_latency = torch.cat((features, latency_values), dim=1)
        # Generate the policy for each block

        #version3
        # latency_expanded = latency.view(-1, 1).expand_as(features)
        # scaled_features = features * latency_expanded
        # features_with_latency = torch.cat((scaled_features, latency), dim=1)

        fc1 = self.fc(features_with_latency)
        combined_input_fc2 = torch.cat((fc1, latency_values), dim=1)
        fc2 = self.fc2(torch.relu(combined_input_fc2))
        combined_input_fc3 = torch.cat((fc2, latency_values), dim=1)
        fc3 = self.fc3(torch.relu(combined_input_fc3))
        policy = torch.sigmoid(fc3)
        # out_temp = fc3.view(-1, 8, 2)
        # policy = torch.softmax(out_temp, dim=-1)
        return policy#[:,:,0]
