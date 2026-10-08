"""Learned tuning diagnostic model (Section 4.1).

Architecture:
  - two independent R-GCNs embed the query/workload and the index
    configuration on the schema graph (representation layer);
  - vertex embedding = concat(last hidden state, input features);
  - rule-based readout: sum pooling over the referenced nodes;
  - workload embedding = weighted average of query embeddings;
  - prediction layer: two-layer fully-connected network on
    concat(e_w, e_i, memory_budget, improvement_threshold) -> logit.

Graphs are stored as flat (src, dst, rel) edge arrays so that a layer's
message passing is a single batched matmul plus a per-relation normalized
scatter-add, regardless of the number of relation types.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .featurizer import GraphSample
from .schema_graph import NUM_RELATIONS


class RGCNLayer(nn.Module):
    """h_i' = relu( sum_r sum_{j in N_i^r} (1/|N_i^r|) W_r h_j  +  W_0 h_i )."""

    def __init__(self, in_dim: int, out_dim: int, num_relations: int = NUM_RELATIONS):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_relations, in_dim, out_dim))
        for r in range(num_relations):
            nn.init.xavier_uniform_(self.weight[r])
        self.self_loop = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, edges: tuple[torch.Tensor, ...]) -> torch.Tensor:
        src, dst, rel = edges
        out = self.self_loop(h)
        if src.numel():
            msg = torch.bmm(h[src].unsqueeze(1), self.weight[rel]).squeeze(1)
            # per-(node, relation) normalization constant c_{i,r} = |N_i^r|
            flat = dst * NUM_RELATIONS + rel
            deg = torch.zeros(h.size(0) * NUM_RELATIONS, device=h.device)
            deg.index_add_(0, flat, torch.ones_like(flat, dtype=h.dtype))
            msg = msg / deg[flat].clamp(min=1.0).unsqueeze(1)
            agg = torch.zeros_like(out).index_add_(0, dst, msg)
            out = out + agg
        return F.relu(out)


def _to_edges(edges: list[tuple[int, int, int]]) -> tuple[torch.Tensor, ...]:
    """(src, rel, dst) tuples -> (src, dst, rel) tensors."""
    if not edges:
        z = torch.empty(0, dtype=torch.long)
        return z, z, z
    a = torch.tensor(edges, dtype=torch.long)
    return a[:, 0], a[:, 2], a[:, 1]


def _cached(sample: GraphSample) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    if "h0" not in sample.torch_cache:
        sample.torch_cache["h0"] = torch.from_numpy(
            np.asarray(sample.features, dtype=np.float32)
        )
        sample.torch_cache["edges"] = _to_edges(sample.edges)
    return sample.torch_cache["h0"], sample.torch_cache["edges"]


class RGCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList(
            RGCNLayer(dims[i], dims[i + 1]) for i in range(num_layers)
        )
        self.out_dim = hidden_dim + in_dim  # concat(h_last, h_0)

    def forward(self, sample: GraphSample) -> torch.Tensor:
        h0, edges = _cached(sample)
        h = h0
        for layer in self.layers:
            h = layer(h, edges)
        e = torch.cat([h, h0], dim=1)
        if sample.readout:
            if "readout" not in sample.torch_cache:
                sample.torch_cache["readout"] = torch.tensor(
                    sample.readout, dtype=torch.long
                )
            return e[sample.torch_cache["readout"]].sum(dim=0)
        return torch.zeros(self.out_dim)


class TuningDiagnostic(nn.Module):
    """End-to-end model: representation layers + prediction layer."""

    def __init__(
        self,
        query_feature_dim: int,
        index_feature_dim: int,
        query_hidden: int = 16,
        index_hidden: int = 8,
        clf_hidden: int = 32,
        num_layers: int = 2,
    ):
        super().__init__()
        self.query_encoder = RGCNEncoder(query_feature_dim, query_hidden, num_layers)
        self.index_encoder = RGCNEncoder(index_feature_dim, index_hidden, num_layers)
        in_dim = self.query_encoder.out_dim + self.index_encoder.out_dim + 2
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, clf_hidden),
            nn.ReLU(),
            nn.Linear(clf_hidden, 1),
        )

    def encode_workload(
        self, queries: list[GraphSample], query_weights: list[float]
    ) -> torch.Tensor:
        """Embed a workload: R-GCN over one disconnected big graph that stacks
        all query graphs, then weighted average of the per-query embeddings."""
        n = len(queries)
        if n == 1:
            return self.query_encoder(queries[0])  # normalized weight is 1.0

        feats, srcs, dsts, rels, read_idx, seg_idx = [], [], [], [], [], []
        offset = 0
        for k, q in enumerate(queries):
            h0, (src, dst, rel) = _cached(q)
            n_nodes = h0.size(0)
            feats.append(h0)
            srcs.append(src + offset)
            dsts.append(dst + offset)
            rels.append(rel)
            for i in q.readout:
                read_idx.append(i + offset)
                seg_idx.append(k)
            offset += n_nodes

        h0 = torch.cat(feats)
        edges = (torch.cat(srcs), torch.cat(dsts), torch.cat(rels))
        h = h0
        for layer in self.query_encoder.layers:
            h = layer(h, edges)
        e = torch.cat([h, h0], dim=1)
        pooled = torch.zeros(n, e.size(1))
        if read_idx:
            pooled.index_add_(
                0, torch.tensor(seg_idx), e[torch.tensor(read_idx)]
            )
        w = torch.tensor(query_weights, dtype=torch.float32)
        w = w / w.sum()
        return (pooled * w.unsqueeze(1)).sum(dim=0)

    def forward(
        self,
        queries: list[GraphSample],
        query_weights: list[float],
        index_config: GraphSample,
        memory_budget: float,
        improvement_threshold: float,
    ) -> torch.Tensor:
        e_w = self.encode_workload(queries, query_weights)
        e_i = self.index_encoder(index_config)
        scalars = torch.tensor(
            [memory_budget, improvement_threshold], dtype=torch.float32
        )
        x = torch.cat([e_w, e_i, scalars])
        return self.classifier(x).squeeze(-1)
