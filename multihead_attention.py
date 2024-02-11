import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = input_size // num_heads

        # Linear layers for query, key, value, and output projections
        self.W_q = nn.Linear(input_size, input_size)
        self.W_k = nn.Linear(input_size, input_size)
        self.W_v = nn.Linear(input_size, input_size)
        # self.W_o = nn.Linear(input_size, input_size)

    def forward(self, x):
        # Split the input features into multiple heads
        q = self.W_q(x).view(x.size(0), -1, self.num_heads, self.head_size)
        k = self.W_k(x).view(x.size(0), -1, self.num_heads, self.head_size)
        v = self.W_v(x).view(x.size(0), -1, self.num_heads, self.head_size)

        # Transpose for attention calculation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Calculate attention weights
        attn_weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / self.head_size**0.5, dim=-1)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)

        # Transpose and concatenate the outputs from multiple heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(x.size(0), -1, self.num_heads * self.head_size)

        # Final output through linear layer
        # output = self.W_o(attn_output)

        return attn_output

