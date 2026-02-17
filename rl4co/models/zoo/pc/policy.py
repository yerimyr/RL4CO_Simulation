from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
from rl4co.models.nn.env_embeddings.context import EnvContext
from rl4co.utils.ops import gather_by_index
from rl4co.models.zoo.am.policy import AttentionModelPolicy


class PCInitEmbedding(nn.Module):
    """Initial embedding for PC instances.

    Expects td["node_features"] of shape [B, N, F].
    """

    def __init__(self, node_feat_dim: int, embed_dim: int, linear_bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(node_feat_dim, embed_dim, bias=linear_bias)

    def forward(self, td: TensorDict):
        return self.proj(td["node_features"].float())


class PCContextEmbedding(EnvContext):
    """Context embedding for PC.

    We concatenate:
      - embedding of current_node
      - mean embedding of nodes in the current open group (or zeros if empty)

    Then project to embed_dim.
    """

    def __init__(self, embed_dim: int, linear_bias: bool = False):
        super().__init__(embed_dim=embed_dim, step_context_dim=2 * embed_dim, linear_bias=linear_bias)

    def _state_embedding(self, embeddings, td):
        # embeddings: [B,N,E]
        open_group = td["open_group"].bool()  # [B,N]
        denom = open_group.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean_emb = (embeddings * open_group.unsqueeze(-1).float()).sum(dim=-2) / denom
        # if empty, mean_emb becomes 0 since numerator is 0
        return mean_emb


def make_pc_policy(
    node_feat_dim: int,
    embed_dim: int = 128,
    num_encoder_layers: int = 3,
    num_heads: int = 8,
    **kwargs,
) -> AttentionModelPolicy:
    """Factory for an AttentionModelPolicy configured for PartConsolidationEnv."""

    init_embedding = PCInitEmbedding(node_feat_dim=node_feat_dim, embed_dim=embed_dim)
    context_embedding = PCContextEmbedding(embed_dim=embed_dim)
    dynamic_embedding = StaticEmbedding(embed_dim)

    return AttentionModelPolicy(
        env_name="pc",
        embed_dim=embed_dim,
        num_encoder_layers=num_encoder_layers,
        num_heads=num_heads,
        init_embedding=init_embedding,
        context_embedding=context_embedding,
        dynamic_embedding=dynamic_embedding,
        **kwargs,
    )  