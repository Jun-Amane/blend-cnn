import torch
import torch.nn as nn
import torchvision.models as models


class textResNet(nn.Module):
    def __init__(self, num_classes, dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)

        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(in_channels=10, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet_conv = nn.Sequential(*list(resnet.children())[:-2])
        self.post_conv = nn.Sequential(nn.Conv1d(512, 256, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.Dropout(p=dropout)
                                    )

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):

        features = self.resnet_conv(x)
        features = self.post_conv(
            features.squeeze())
        features = nn.functional.max_pool1d(features, kernel_size=features.size(2)).squeeze(2)
        output = self.fc(features)

        return output


