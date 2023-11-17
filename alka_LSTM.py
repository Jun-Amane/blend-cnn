import torch
import torch.nn as nn


class AlkaLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, tagset_size):
        super(AlkaLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        # batch_first=Trueが大事！
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sentence):
        # embeds.size() = (batch_size × len(sentence) × embedding_dim)
        _, lstm_out = self.lstm(sentence)
        # lstm_out[0].size() = (1 × batch_size × hidden_dim)
        tag_space = self.hidden2tag(lstm_out[0])
        # tag_space.size() = (1 × batch_size × tagset_size)

        # (batch_size × tagset_size)にするためにsqueeze()する
        tag_scores = self.softmax(tag_space.squeeze())
        # tag_scores.size() = (batch_size × tagset_size)

        return tag_scores
