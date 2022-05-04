import math
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import torch.nn.functional as F


class Linear(nn.Linear):
    
    def __init__(self, in_features: int, out_features: int, init_scale=np.sqrt(2), init_bias=0.0, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        nn.init.orthogonal_(self.weight, gain=init_scale)
        if bias:
            self.bias.data.fill_(init_bias)

class Conv1d(nn.Conv1d):

    def __init__(self, in_channels: int, out_channels: int, kernel_size=1, stride=1, init_scale=np.sqrt(2), init_bias=0.0, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, bias=bias, device=device, dtype=dtype)
        nn.init.orthogonal_(self.weight, gain=init_scale)
        if bias:
            self.bias.data.fill_(init_bias)

class GatConv(MessagePassing):

    def __init__(self, in_channels, out_channels, edge_channels, negative_slope=0.2, dropout=0):
        super(GatConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.fc = Linear(in_channels, out_channels)
        self.attn = Linear(2 * out_channels + edge_channels, out_channels)

    def forward(self, x, edge_index, edge_attr, size=None):
        x = self.fc(x)
        return self.propagate(edge_index, size=size, x=x, edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        x = torch.cat([x_i, x_j, edge_attr], dim=-1)
        alpha = self.attn(x)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha

    def update(self, aggr_out):
        return aggr_out


class Attention(nn.Module):

    def __init__(self, n_heads, cat, input_dim, hidden_dim):
        super().__init__()
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.head_dim = self.hidden_dim / self.n_heads
        self.norm = 1 / math.sqrt(self.head_dim)
        self.w = Linear(input_dim * cat, hidden_dim, bias=False)
        self.k = Linear(input_dim, hidden_dim, bias=False)
        self.v = Linear(input_dim, hidden_dim, bias=False)
        self.fc = Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, state_t, context, mask):
        batch_size, n_nodes, _ = context.size()
        Q = self.w(state_t).view(batch_size, 1, self.n_heads, -1)
        K = self.k(context).view(batch_size, n_nodes, self.n_heads, -1)
        V = self.v(context).view(batch_size, n_nodes, self.n_heads, -1)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        compatibility = self.norm * torch.matmul(Q, K.transpose(2, 3))
        compatibility = compatibility.squeeze(2)
        mask = mask.unsqueeze(1).expand_as(compatibility)
        u_i = compatibility.masked_fill(mask.bool(), float("-inf"))
        scores = F.softmax(u_i, dim=-1)
        scores = scores.unsqueeze(2)
        out_put = torch.matmul(scores, V)
        out_put = out_put.squeeze(2).view(batch_size, self.hidden_dim)
        out_put = self.fc(out_put)
        return out_put
        
        
class ProbAttention(nn.Module):

    def __init__(self, n_heads, input_dim, hidden_dim):
        super().__init__()
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.norm = 1 / math.sqrt(hidden_dim)
        self.k = Linear(input_dim, hidden_dim, bias=False)
        self.mhalayer = Attention(n_heads, 1, input_dim, hidden_dim)

    def forward(self, state_t, context, mask):
        x = self.mhalayer(state_t, context, mask)
        batch_size, n_nodes, _ = context.size()
        Q = x.view(batch_size, 1, -1)
        K = self.k(context).view(batch_size, n_nodes, -1)
        compatibility = self.norm * torch.matmul(Q, K.transpose(1, 2))
        compatibility = compatibility.squeeze(1)
        x = torch.tanh(compatibility)
        x = x.masked_fill(mask.bool(), float("-inf"))
        scores = F.softmax(x, dim=-1)
        return scores

