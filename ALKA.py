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

        # multi-head attention, in=out=512+512
        self.attention = MultiHeadAttention(1024, num_heads)
        self.layer_norm = nn.LayerNorm(1024)

        # feed forward network
        self.feed_forward_network = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes)
        )


    def forward(self, image, captions):
        # Image feature extracting
        image_features = self.image_model(image)
        image_features = image_features.view(image_features.size(0), -1)

        # Text feature extracting
        text_features = self.text_cnn(captions)

        # MM fusion by multi-head attention
        concat_features = torch.cat((image_features, text_features), dim=1)
        attn_out = self.attention(concat_features).squeeze()
        # with residual connection
        attn_out = self.layer_norm(attn_out + concat_features)
        # FFN
        ffn_out = self.feed_forward_network(attn_out)

        return ffn_out


# num_classes = 102
# model = ALKA(num_classes=num_classes)
# print(model)
