import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchtext.vocab import GloVe


class AlkaTextCNN(nn.Module):
    def __init__(self,
                 embed_dim=300,
                 filter_sizes=None,
                 num_filters=None):

        super(AlkaTextCNN, self).__init__()
        if num_filters is None:
            num_filters = [384, 384, 384, 384]
        if filter_sizes is None:
            filter_sizes = [3, 4, 5, 6]

        # Embedding layer

        # Conv Network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])


    def forward(self, input_ids):

        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = input_ids.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_hidden = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)

        return x_hidden
