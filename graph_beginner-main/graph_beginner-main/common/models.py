from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.utils import scatter


class GNNEncoder(nn.Module):
    """A small, uniform GCN/GAT/GraphSAGE/GIN node encoder."""

    def __init__(self, kind: str, in_channels: int, hidden_channels: int, layers: int):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be at least 1")
        kind = kind.lower()
        if kind not in {"gcn", "gat", "sage", "gin"}:
            raise ValueError(f"unsupported GNN model: {kind}")
        self.kind = kind
        self.convs = nn.ModuleList()
        channels = in_channels
        for _ in range(layers):
            if kind == "gcn":
                conv = GCNConv(channels, hidden_channels)
            elif kind == "gat":
                conv = GATConv(channels, hidden_channels, heads=1)
            elif kind == "sage":
                conv = SAGEConv(channels, hidden_channels)
            else:
                mlp = nn.Sequential(
                    nn.Linear(channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )
                conv = GINConv(mlp)
            self.convs.append(conv)
            channels = hidden_channels

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = x.float()
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return x


class NodeClassifier(nn.Module):
    def __init__(
        self,
        kind: str,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        layers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.encoder = GNNEncoder(kind, in_channels, hidden_channels, layers)
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x, edge_index)
        return self.classifier(F.dropout(x, p=self.dropout, training=self.training))


class GraphClassifier(nn.Module):
    def __init__(
        self,
        kind: str,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        layers: int,
        pooling: str,
        dropout: float = 0.5,
    ):
        super().__init__()
        if pooling not in {"avg", "max", "min"}:
            raise ValueError(f"unsupported pooling method: {pooling}")
        self.encoder = GNNEncoder(kind, in_channels, hidden_channels, layers)
        self.pooling = pooling
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        x = self.encoder(x, edge_index)
        if self.pooling == "avg":
            pooled = global_mean_pool(x, batch)
        elif self.pooling == "max":
            pooled = global_max_pool(x, batch)
        else:
            pooled = scatter(x, batch, dim=0, reduce="min")
        return self.classifier(F.dropout(pooled, p=self.dropout, training=self.training))


class KnowledgeGraphScorer(nn.Module):
    """TransE, RotatE and a compact ConvE-style scorer."""

    def __init__(self, kind: str, entities: int, relations: int, dim: int):
        super().__init__()
        kind = kind.lower()
        if kind not in {"transe", "rotate", "conve"}:
            raise ValueError(f"unsupported knowledge graph model: {kind}")
        if dim < 4:
            raise ValueError("embedding dimension must be at least 4")
        if kind == "rotate" and dim % 2:
            raise ValueError("RotatE requires an even embedding dimension")
        self.kind = kind
        self.dim = dim
        self.entity = nn.Embedding(entities, dim)
        relation_dim = dim // 2 if kind == "rotate" else dim
        self.relation = nn.Embedding(relations, relation_dim)
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)
        if kind == "conve":
            self.height = math.isqrt(dim)
            self.width = math.ceil(dim / self.height)
            self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.projection = nn.Linear(16 * 2 * self.height * self.width, dim)
            self.dropout = nn.Dropout(0.2)

    def _conve_query(self, heads: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        h = F.pad(heads, (0, self.height * self.width - self.dim))
        r = F.pad(relations, (0, self.height * self.width - self.dim))
        h = h.view(-1, 1, self.height, self.width)
        r = r.view(-1, 1, self.height, self.width)
        stacked = torch.cat((h, r), dim=2)
        hidden = F.relu(self.conv(stacked)).flatten(start_dim=1)
        return self.dropout(F.relu(self.projection(hidden)))

    def score(self, heads: torch.Tensor, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        h = self.entity(heads)
        t = self.entity(tails)
        r = self.relation(relations)
        if self.kind == "transe":
            return -(h + r - t).abs().sum(dim=-1)
        if self.kind == "rotate":
            h_re, h_im = h.chunk(2, dim=-1)
            t_re, t_im = t.chunk(2, dim=-1)
            phase = r
            r_re, r_im = torch.cos(phase), torch.sin(phase)
            distance = torch.sqrt(
                (h_re * r_re - h_im * r_im - t_re).square()
                + (h_re * r_im + h_im * r_re - t_im).square()
                + 1e-12
            )
            return -distance.sum(dim=-1)
        query = self._conve_query(h, r)
        return (query * t).sum(dim=-1)

    def score_all_tails(self, heads: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        h = self.entity(heads)
        r = self.relation(relations)
        candidates = self.entity.weight
        if self.kind == "transe":
            return -(h[:, None, :] + r[:, None, :] - candidates[None, :, :]).abs().sum(dim=-1)
        if self.kind == "rotate":
            h_re, h_im = h.chunk(2, dim=-1)
            t_re, t_im = candidates.chunk(2, dim=-1)
            r_re, r_im = torch.cos(r), torch.sin(r)
            re = h_re[:, None, :] * r_re[:, None, :] - h_im[:, None, :] * r_im[:, None, :]
            im = h_re[:, None, :] * r_im[:, None, :] + h_im[:, None, :] * r_re[:, None, :]
            distance = torch.sqrt(
                (re - t_re[None, :, :]).square() + (im - t_im[None, :, :]).square() + 1e-12
            )
            return -distance.sum(dim=-1)
        query = self._conve_query(h, r)
        return query @ candidates.t()
