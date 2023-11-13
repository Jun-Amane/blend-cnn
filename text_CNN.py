import torch
import torch.nn as nn
import torchvision.models as models


class textResNet(nn.Module):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pre_conv = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        resnet = models.resnet18(weights=None)
        self.resnet_conv = nn.Sequential(*list(resnet.children())[:-2])

        self.conv1d = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):

        x = self.pre_conv(x)
        features = self.resnet_conv(x)
        features = self.conv1d(
            features.squeeze())
        features = nn.functional.relu(features)
        features = nn.functional.max_pool1d(features, kernel_size=features.size(2)).squeeze(2)
        output = self.fc(features)

        return output


