import torch
import torch.nn as nn
import torchvision
from text_CNN import AlkaTextCNN
from multihead_attention import MultiHeadAttention


class ALKA(nn.Module):
    def __init__(self, num_classes, pretrained_embedding, dropout=0.5, num_heads=8, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # image CNN, in_channel=3, out=512
        self.image_model = torchvision.models.resnet18(weights=None)
        self.image_model = nn.Sequential(*list(self.image_model.children())[:-1])

        # text CNN, in_channel=embed_dim=pretrained_embedding.shape, out=512
        self.text_cnn = AlkaTextCNN(pretrained_embedding=pretrained_embedding)

        # multi-head attention, in=out=512
        self.attn_fusion = MultiHeadAttention(512, num_heads)

        # MM fusion, in=512(img)+512(text), out=num_classes=102
        self.fc_fusion = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512 + 512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, captions):
        # Image feature extracting
        image_features = self.image_model(image)
        image_features = image_features.view(image_features.size(0), -1)

        # Text feature extracting
        # Input B 10 128(padding) 768
        text_features = self.text_cnn(captions)

        text_attn = self.attn_fusion(text_features)
        img_attn = self.attn_fusion(image_features)

        # MM fusion
        fusion_input = torch.cat((img_attn, text_attn), dim=1)
        output = self.fc_fusion(fusion_input)

        return output


# num_classes = 102
# model = ALKA(num_classes=num_classes)
# print(model)
