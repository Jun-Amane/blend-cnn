import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from text_CNN import AlkaTextCNN
from multihead_attention import MultiHeadAttention
from transformers import BertModel


class ModifiedmCNN(nn.Module):
    def __init__(self, drop_out, drop_out2, embedding_matrix, max_sequence_length, embedding_dim,
                 num_filters, filter_sizes):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        self.vgg = torchvision.models.vgg16(weights='DEFAULT')
        self.vgg.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096)
        )

        self.conv_x = nn.Conv2d(256, 256, (14, 1))
        self.dropout_x = nn.Dropout(drop_out2)
        self.maxpool_x = nn.MaxPool2d((2, 1))

        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, num_filters, kernel_size=size) for size in filter_sizes])
        self.dropout_convs = nn.ModuleList([nn.Dropout(drop_out2) for _ in filter_sizes])
        self.conva = nn.Conv2d(256, 512, kernel_size=(5, 1))
        self.maxa = nn.MaxPool2d((2, 1))

        self.fc1 = nn.Linear(512, 102)
        self.dropout_z = nn.Dropout(drop_out)
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.flatten = nn.Flatten()

    def forward(self, inputs1, inputs2):
        inputs1 = self.vgg(inputs1)
        x = inputs1.view(-1, 256, 16, 1)
        x = F.relu(self.conv_x(x))
        x = self.dropout_x(x)
        max_x = self.maxpool_x(x).squeeze(dim=3)

        y = self.embedding(inputs2).float()
        y = y.permute(0, 2, 1)
        conved = [F.relu(conv(y)) for conv in self.convs]
        conved = [dropout(conv) for conv, dropout in zip(conved, self.dropout_convs)]
        pooled = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                  for x_conv in conved]

        concatenated = torch.stack([max_x, pooled[0], max_x, pooled[1], max_x, pooled[2]], dim=2)
        hidden = self.flatten(self.maxa(self.conva(concatenated)))
        output = self.fc1(hidden)

        return output


class GalloEtAls(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.incep = torchvision.models.inception_v3(weights='DEFAULT')
        self.incep.aux_logits = False
        del self.incep.dropout
        del self.incep.fc
        self.incep.add_module('pooling', nn.AvgPool2d((8, 8)))
        self.incep.add_module('dropout', nn.Dropout(0.4))
        self.incep.add_module('flatten', nn.Flatten())
        self.incep.add_module('fc', nn.Linear(2048, 101))

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(input_size=768, hidden_size=128, batch_first=True)
        self.text_seq = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.Dropout(0.5),
            nn.Linear(256, 101),
        )

        self.fuse_seq = nn.Sequential(
            nn.Linear(202, 256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 102),
        )

    def forward(self, inputs1, inputs2, attn_msk):
        image_features = self.incep(inputs1)

        _, bert_out = self.bert(input_ids=inputs2, attention_mask=attn_msk, return_dict=False)
        lstm_out, _ = self.lstm(bert_out)
        text_features = self.text_seq(lstm_out)

        fused_features = torch.concat((image_features, text_features), dim=1)
        out = self.fuse_seq(fused_features)
        return out
