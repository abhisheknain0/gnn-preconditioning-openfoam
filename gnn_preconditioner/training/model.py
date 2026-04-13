"""MeshGraphNet backbone reused for the new cell-volume stack."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from . import config


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, hidden_layers: int = 2, layer_norm: bool = True):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class EdgeBlock(nn.Module):
    def __init__(self, hidden_dim: int, mlp_layers: int):
        super().__init__()
        self.mlp = MLP(3 * hidden_dim, hidden_dim, hidden_dim, mlp_layers, layer_norm=True)

    def forward(self, edge_feat: Tensor, node_feat: Tensor, edge_index: Tensor) -> Tensor:
        src, dst = edge_index[0], edge_index[1]
        inp = torch.cat([edge_feat, node_feat[src], node_feat[dst]], dim=-1)
        return self.mlp(inp) + edge_feat


class NodeBlock(nn.Module):
    def __init__(self, hidden_dim: int, mlp_layers: int, aggregation: str = "sum"):
        super().__init__()
        self.aggregation = aggregation
        self.mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim, mlp_layers, layer_norm=True)

    def forward(self, node_feat: Tensor, edge_feat: Tensor, edge_index: Tensor) -> Tensor:
        dst = edge_index[1]
        agg = torch.zeros((node_feat.size(0), edge_feat.size(1)), device=edge_feat.device, dtype=edge_feat.dtype)
        agg.index_add_(0, dst, edge_feat)
        if self.aggregation == "mean":
            counts = torch.zeros((node_feat.size(0),), device=edge_feat.device, dtype=edge_feat.dtype)
            counts.index_add_(0, dst, torch.ones((dst.numel(),), device=edge_feat.device, dtype=edge_feat.dtype))
            agg = agg / counts.clamp_min(1.0).unsqueeze(-1)
        elif self.aggregation != "sum":
            raise ValueError(f"Unsupported aggregation={self.aggregation!r}")
        inp = torch.cat([node_feat, agg], dim=-1)
        return self.mlp(inp) + node_feat


class MeshGraphNet(nn.Module):
    def __init__(
        self,
        input_dim_nodes: int = config.NUM_NODE_INPUT,
        input_dim_edges: int = config.NUM_EDGE_INPUT,
        output_dim: int = config.NUM_OUTPUT,
        hidden_dim: int = config.HIDDEN_DIM,
        num_mp_steps: int = config.MESSAGE_PASSING_STEPS,
        mlp_layers: int = config.MLP_LAYERS,
        aggregation: str = config.AGGREGATION,
    ):
        super().__init__()
        self.node_encoder = MLP(input_dim_nodes, hidden_dim, hidden_dim, mlp_layers, layer_norm=True)
        self.edge_encoder = MLP(input_dim_edges, hidden_dim, hidden_dim, mlp_layers, layer_norm=True)
        self.edge_blocks = nn.ModuleList([EdgeBlock(hidden_dim, mlp_layers) for _ in range(num_mp_steps)])
        self.node_blocks = nn.ModuleList([NodeBlock(hidden_dim, mlp_layers, aggregation) for _ in range(num_mp_steps)])
        self.node_decoder = MLP(hidden_dim, output_dim, hidden_dim, mlp_layers, layer_norm=False)

    def forward(self, node_features: Tensor, edge_features: Tensor, edge_index: Tensor) -> Tensor:
        nodes = self.node_encoder(node_features)
        edges = self.edge_encoder(edge_features)
        for edge_block, node_block in zip(self.edge_blocks, self.node_blocks):
            edges = edge_block(edges, nodes, edge_index)
            nodes = node_block(nodes, edges, edge_index)
        return self.node_decoder(nodes)
