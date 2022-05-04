import torch
import torch.nn as nn
from .layers import GatConv, Linear


class Encoder(nn.Module):

    def __init__(self, input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim):
        super().__init__()
        self.hidden_node_dim = hidden_node_dim
        self.fc_node = Linear(input_node_dim, hidden_node_dim)
        self.bn = nn.BatchNorm1d(hidden_node_dim)
        self.be = nn.BatchNorm1d(hidden_edge_dim)
        self.fc_edge = Linear(input_edge_dim, hidden_edge_dim)
        self.conv1 = GatConv(hidden_node_dim, hidden_node_dim, hidden_edge_dim)
        self.conv2 = GatConv(hidden_node_dim, hidden_node_dim, hidden_edge_dim)

    def forward(self, data):
        features = torch.cat([data.pos, data.x], -1)
        features = self.bn(self.fc_node(features))
        edge_attr = self.be(self.fc_edge(data.edge_attr))
        features = features + self.conv1(features, data.edge_index, edge_attr)
        features = features + self.conv2(features, data.edge_index, edge_attr)
        ptr = data.ptr.clone()
        embs = []
        prev_id = ptr[0]
        for id in ptr[1:]:
            emb = features[prev_id:id]
            embs.append(emb)
            prev_id = id
        return embs
