import torch
import torch.nn as nn


class AlkaLSTM(nn.Module):
    def __init__(self, hidden_dim, tagset_size, vocab_size, embedding_dim=256):
        super(AlkaLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # batch_first=Trueが大事！
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, tokens, attention_mask):
        # tokens (captions): (batch_size, 10, 128)
        # masks also
        embedded_tokens = self.embedding(tokens)
        # embedded_tokens: (batch_size, 10, 128, embedding_dim)
        # attention_mask.unsqueeze(3): (batch_size, 10, 128, 1)
        # product: (batch_size, 10, 128)
        # embedded_tokens = embedded_tokens * attention_mask.unsqueeze(3)

        # mean: (batchsize, 128, embedding_dim)
        embedded_tokens = torch.mean(embedded_tokens, dim=1)  # also try torch.sum()

        # embeds.size() = (batch_size × len(sentence) × embedding_dim)
        _, lstm_out = self.lstm(embedded_tokens)
        # lstm_out[0].size() = (1 × batch_size × hidden_dim)
        tag_space = self.hidden2tag(lstm_out[0])
        # tag_space.size() = (1 × batch_size × tagset_size)

        # (batch_size × tagset_size)にするためにsqueeze()する
        tag_scores = self.softmax(tag_space.squeeze())
        # tag_scores.size() = (batch_size × tagset_size)

        return tag_scores
