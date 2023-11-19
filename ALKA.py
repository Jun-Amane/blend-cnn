import torch
import torch.nn as nn
import torchvision
# from text_CNN import textResNet
from alka_LSTM import AlkaLSTM
from transformers import BertModel, BertTokenizer, BertForSequenceClassification


class ALKA(nn.Module):
    def __init__(self, num_classes, vocab_size, dropout=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # image CNN, in_channel=3, out=512
        self.image_model = torchvision.models.resnet18(weights=None)
        self.image_model = nn.Sequential(*list(self.image_model.children())[:-1])

        # For text embedding
        # self.embedding = nn.EmbeddingBag(50000, 768, sparse=False)

        # text CNN, in_channel=1, out=num_classes=512
        # self.text_cnn = textResNet(num_classes=512, dropout=dropout)
        # self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.text_lstm = AlkaLSTM(hidden_dim=256, tagset_size=512, vocab_size=vocab_size)

        # self.bert = BertModel.from_pretrained("bert-base-uncased")
        # self.linear_projection = nn.Linear(768, 512)

        # MM fusion
        self.fc_fusion = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, captions):
        # # Image feature extracting
        # image_features = self.image_model(image)
        # image_features = image_features.view(image_features.size(0), -1)
        # image_features = torch.nn.functional.log_softmax(image_features, dim=1)

        # Text feature extracting
        # Input B 10 128(padding) 768

        # captions_list = []
        # for i in range(captions.size(1)):
        #     current_dimension_data = captions[:, i, :]
        #     lstm_out = self.text_lstm(current_dimension_data)
        #     # embedded = self.embedding(current_dimension_data)
        #     # embedded = embedded.unsqueeze(
        #     #     1)  # to B H=1 W
        #     captions_list.append(lstm_out)
        # # stack to B C=10 H W
        # text_output = torch.stack(captions_list, dim=1)

        # text_output = self.text_cnn(captions_output)

        captions_list = []
        for i in range(captions.size(1)):
            current_dimension_data = captions[:, i, :]
            bert_output = self.text_lstm(current_dimension_data)
            captions_list.append(bert_output)
        text_output = torch.stack(captions_list, dim=1)

        # mixing the dim 1, to B C H W
        text_pooled = torch.mean(text_output, dim=1)
        # text_features = self.linear_projection(text_pooled)
        # text_pooled = text_pooled.unsqueeze(0)

        # captions: (batch_size, 10, 128)
        # masks also

        # MM fusion
        # fusion_input = torch.cat((image_features, text_features), dim=1)
        # output = self.fc_fusion(text_pooled)

        return text_pooled


# num_classes = 102
# model = ALKA(num_classes=num_classes)
# print(model)
