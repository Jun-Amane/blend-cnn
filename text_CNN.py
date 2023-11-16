import torch
import torch.nn as nn
import torchvision.models as models


class textResNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pre_conv = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        resnet = models.resnet18(weights=None)
        self.resnet_conv = resnet
        self.resnet_conv.fc = nn.Linear(in_features=512, out_features=102)

    def forward(self, x):

        x = self.pre_conv(x)
        output = self.resnet_conv(x)

        return output


