import torch
import torch.nn as nn
import torchvision
from text_CNN import AlkaTextCNN
from multihead_attention import MultiHeadAttention


class ALKA(nn.Module):
    def __init__(self, num_classes, pretrained_embedding, dropout=0.5, num_heads=8, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # image CNN, in_channel=3, out=1536
        self.image_model = torchvision.models.efficientnet_b3(weights='IMAGENET1K_V1')
        self.image_model = nn.Sequential(*list(self.image_model.children())[:-1])

        # text CNN, in_channel=embed_dim=pretrained_embedding.shape, out=1536
        self.text_cnn = AlkaTextCNN(pretrained_embedding=pretrained_embedding)

        # multi-head attention, in=out=1536
        # self.attention = MultiHeadAttention(1536, num_heads)
        # self.attention = nn.MultiheadAttention(embed_dim=1536, num_heads=8, batch_first=True)
        # self.layer_norm = nn.LayerNorm(1536)
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7)

        # average pooling, in=(b, 2, 1536) out=(b, 1, 768)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()

        self.img_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1536, 102)
        )

        self.text_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1536 * 4, 102)
        )

        # feed forward network
        self.feed_forward_network = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(765, num_classes)
        )


    def forward(self, image, captions):
        # Image feature extracting
        # Output shape: (b, 1536)
        image_features = self.image_model(image)
        image_features = image_features.view(image_features.size(0), -1)

        # Text feature extracting
        # Output shape: (b, 1536)
        text_features = self.text_cnn(captions)

        # Stack two features
        # Output shape: (b, 2, 1536)
        stack_features = torch.stack([image_features, text_features[0], image_features, text_features[1],
                                      image_features, text_features[2], image_features, text_features[3]], dim=1)
        stack_features = stack_features.unsqueeze(1)

        # Learning with Multihead Self Attention
        # Output shape: (b, 2, 1536)
        # attn_out, _ = self.attention(stack_features, stack_features, stack_features, need_weights=False)
        # with residual connection
        # attn_out = self.layer_norm(attn_out + stack_features)
        conv_out = self.conv(stack_features)


        # Average Pooling
        # Output shape (after squeeze): (b, 768)
        pool_out = self.pool(conv_out)
        flat_out=  self.flatten(pool_out)


        # FFN
        # Output shape: (b, num_classes)
        ffn_out = self.feed_forward_network(flat_out)

        img_out = self.img_classifier(image_features)
        text_out = self.text_classifier(torch.concat(text_features, dim=1))

        return ffn_out, img_out, text_out


# num_classes = 102
# model = ALKA(num_classes=num_classes)
# print(model)
