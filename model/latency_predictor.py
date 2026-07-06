import torch
import torch.nn as nn

class Predictor(nn.Module):
    """"""

    def __init__(self,num_blocks):
        super(Predictor, self).__init__()
        self.pred = nn.Sequential(
            nn.Linear(num_blocks, 64, bias=True), ##resnet18 input is 8;
            nn.ReLU(inplace=True),
            nn.Linear(64, 32, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1, bias=True)
        )

    def forward(self, x):
        # s is the policy
        return self.pred(x)
