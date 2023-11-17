import torch
import torch.nn as nn
import torchvision
# from transformers import BertModel, BertTokenizer
from text_CNN import textResNet


class ALKA(nn.Module):
    def __init__(self, num_classes, dropout=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # image CNN, in_channel=3, out=512
        self.image_model = torchvision.models.resnet18(weights=None)
        self.image_model = nn.Sequential(*list(self.image_model.children())[:-1])

        # For text embedding
        self.embedding = nn.EmbeddingBag(50000, 768, sparse=False)

        # text CNN, in_channel=1, out=num_classes=512
        self.text_cnn = textResNet(num_classes=512, dropout=dropout)

        # MM fusion
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

        # Text feature extracting
        # Input B 10 W
        # Convert to B C=10 H=1 W

        captions_list = []
        for i in range(captions.size(1)):
            current_dimension_data = captions[:, i, :]
            embedded = self.embedding(current_dimension_data)
            embedded = embedded.unsqueeze(
                1)  # to B H=1 W
            captions_list.append(embedded)
        # stack to B C=10 H W
        captions_output = torch.stack(captions_list, dim=1)

        text_output = self.text_cnn(captions_output)

        # mixing the dim 1, to B C H W
        # text_pooled = torch.mean(text_output, dim=1)
        # text_pooled = text_pooled.unsqueeze(0)

        # MM fusion
        fusion_input = torch.cat((image_features.view(image_features.size(0), -1), text_output), dim=1)
        output = self.fc_fusion(fusion_input)

        return output


num_classes = 102
model = ALKA(num_classes=num_classes)
print(model)

