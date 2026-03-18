from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict


class EdgeAwareMessagePassing(nn.Module):
    def __init__(self, emb_dim: int, edge_feat_dim: int):
        super().__init__()
        self.node_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.edge_proj = nn.Linear(edge_feat_dim, emb_dim, bias=False)
        self.weight_proj = nn.Linear(1, emb_dim, bias=False)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, h: torch.Tensor, edge_features: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        hi = self.node_proj(h).unsqueeze(2)
        ej = self.edge_proj(edge_features)
        wij = self.weight_proj(W.unsqueeze(-1))

        msg = torch.tanh(hi + ej + wij)
        agg = msg.mean(dim=2)
        return self.out_proj(agg)


class PCPolicy(nn.Module):
    """
    State-aware edge-aware policy.

    Static inputs:
    - node_features
    - edge_features
    - W

    Dynamic state inputs (updated every step and re-encoded every step):
    - assigned
    - open_group
    - action_mask
    - current open_group_size / build_limit
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        emb_dim: int = 128,
        num_message_passing: int = 3,
    ):
        super().__init__()

        self.base_node_feat_dim = node_feat_dim
        self.dynamic_node_feat_dim = 1 + 1 + 1 + 3  # assigned, open_group, action_mask, open_group_size_ratio(3)
        self.total_node_feat_dim = self.base_node_feat_dim + self.dynamic_node_feat_dim

        self.node_embed = nn.Linear(self.total_node_feat_dim, emb_dim)
        self.layers = nn.ModuleList(
            [EdgeAwareMessagePassing(emb_dim, edge_feat_dim) for _ in range(num_message_passing)]
        )

        self.context_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.key_proj = nn.Linear(emb_dim, emb_dim, bias=False)
        self.logit_bias = nn.Parameter(torch.zeros(1))

    def build_dynamic_node_features(self, td: TensorDict) -> torch.Tensor:
        B, N, _ = td["node_features"].shape

        assigned = td["assigned"].float().unsqueeze(-1)
        open_group = td["open_group"].float().unsqueeze(-1)
        action_mask = td["action_mask"].float().unsqueeze(-1)

        build_limit = td["build_limit"].float().clamp_min(1e-6)
        open_group_size = td["open_group_size"].float()
        open_group_size_ratio = (open_group_size / build_limit).unsqueeze(1).expand(B, N, 3)

        dyn = torch.cat([assigned, open_group, action_mask, open_group_size_ratio], dim=-1)
        return dyn

    def encode(self, td: TensorDict) -> torch.Tensor:
        x_static = td["node_features"].float()
        x_dynamic = self.build_dynamic_node_features(td)
        x = torch.cat([x_static, x_dynamic], dim=-1)

        e = td["edge_features"].float()
        W = td["W"].float()

        h = self.node_embed(x)
        for layer in self.layers:
            h = h + layer(h, e, W)
        return h

    def compute_logits(self, node_emb: torch.Tensor, td: TensorDict) -> torch.Tensor:
        open_group = td["open_group"]
        B, N, E = node_emb.shape

        has_open = open_group.any(dim=-1, keepdim=True)
        denom = open_group.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        group_mean = (node_emb * open_group.float().unsqueeze(-1)).sum(dim=1) / denom

        sep_emb = node_emb[:, 0, :]
        context = torch.where(has_open, group_mean, sep_emb)

        query = self.context_proj(context).unsqueeze(1)
        keys = self.key_proj(node_emb)

        logits = torch.matmul(query, keys.transpose(-1, -2)).squeeze(1)
        logits = logits / (E ** 0.5)
        logits = logits + self.logit_bias
        return logits

    def act(self, td: TensorDict, sample: bool = True):
        node_emb = self.encode(td)
        logits = self.compute_logits(node_emb, td)

        mask = td["action_mask"]
        logits = logits.masked_fill(~mask, -1e9)
        probs = F.softmax(logits, dim=-1)

        if sample:
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            action = torch.argmax(probs, dim=-1)

        logp = torch.log(probs.gather(1, action.view(-1, 1)).clamp_min(1e-12)).squeeze(-1)
        return action, logp, logits

    def forward(self, td: TensorDict, max_steps: int = 1, sample: bool = True):
        actions = []
        logps = []
        for _ in range(max_steps):
            action, logp, _ = self.act(td, sample=sample)
            actions.append(action)
            logps.append(logp)
        return torch.stack(actions, dim=1), torch.stack(logps, dim=1)


def make_pc_policy(**kwargs):
    return PCPolicy(
        node_feat_dim=kwargs.get("node_feat_dim", 8),
        edge_feat_dim=kwargs.get("edge_feat_dim", 6),
        emb_dim=kwargs.get("emb_dim", 128),
        num_message_passing=kwargs.get("num_message_passing", 3),
    )