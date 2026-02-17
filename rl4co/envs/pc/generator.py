from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict import TensorDict


@dataclass
class PCGeneratorParams:
    num_parts: int = 4
    # random graph density for extra edges beyond a spanning tree
    extra_edge_prob: float = 0.35
    # node feature ranges
    material_types: int = 3  # {0,1,2}
    size_low: float = 0.1
    size_high: float = 1.0
    motion_types: int = 2  # {0,1}
    # build size constraint for reward (not a hard mask)
    build_limit: float = 1.6


class PCGenerator:
    """Random instance generator for the Part Consolidation (PC) prototype.

    Outputs a TensorDict with:
      - node_features: [B, N+1, F] (includes node 0 = SEP)
      - compat:        [B, N+1, N+1] (bool, includes SEP fully compatible)
      - material:      [B, N+1] (long, SEP=-1)
      - size:          [B, N+1] (float, SEP=0)
      - motion:        [B, N+1] (long, SEP=-1)

    Notes
    -----
    * compat is generated as an undirected connected random graph over the N parts.
      We then embed it into an (N+1)x(N+1) matrix and make SEP compatible with all.
    * node_features uses: material one-hot (3), size (1), motion (1) => F=5.
    """

    def __init__(self, **kwargs):
        params = PCGeneratorParams(**kwargs)
        self.params = params

        self.num_parts = params.num_parts
        self.num_nodes = self.num_parts + 1  # + SEP
        self.build_limit = params.build_limit

        self.material_types = params.material_types
        self.motion_types = params.motion_types
        self.size_low = params.size_low
        self.size_high = params.size_high
        self.extra_edge_prob = params.extra_edge_prob

        # feature dim: one-hot material (3) + size (1) + motion (1)
        self.node_feat_dim = self.material_types + 1 + 1

    def _random_connected_undirected_graph(self, n: int, batch_size, device):
        """Return adjacency matrix [B,n,n] for a connected undirected random graph."""
        B = batch_size[0] if isinstance(batch_size, (list, tuple, torch.Size)) else batch_size
        adj = torch.zeros((B, n, n), dtype=torch.bool, device=device)

        # build a random spanning tree per batch to ensure connectivity
        for b in range(B):
            # random permutation of nodes
            perm = torch.randperm(n, device=device)
            # connect each new node to one of the previous nodes
            for i in range(1, n):
                u = perm[i].item()
                v = perm[torch.randint(0, i, (1,), device=device).item()].item()
                adj[b, u, v] = True
                adj[b, v, u] = True
            # add extra edges
            rand_mat = torch.rand((n, n), device=device)
            extra = (rand_mat < self.extra_edge_prob) & (~torch.eye(n, dtype=torch.bool, device=device))
            # keep upper triangle, then mirror
            extra = torch.triu(extra, diagonal=1)
            adj[b] |= extra
            adj[b] |= extra.transpose(0, 1)

        # diagonal doesn't matter (we'll handle in compat)
        return adj

    def __call__(self, batch_size, device: str | torch.device | None = None) -> TensorDict:
        if device is None:
            device = torch.device("cpu")
        else:
            device = torch.device(device)

        batch_size = [batch_size] if isinstance(batch_size, int) else list(batch_size)
        B = batch_size[0]
        N = self.num_parts

        # --- raw part attributes (exclude SEP) ---
        material = torch.randint(0, self.material_types, (B, N), device=device)
        motion = torch.randint(0, self.motion_types, (B, N), device=device)
        size = torch.rand((B, N), device=device) * (self.size_high - self.size_low) + self.size_low

        # --- compat graph over parts ---
        adj = self._random_connected_undirected_graph(N, batch_size, device=device)

        # compat for parts: allow grouping only if there is an edge between i and j
        # (hard spatial constraint). Self-compat is True.
        compat_parts = adj.clone()
        compat_parts |= torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)

        # embed into (N+1)x(N+1) with SEP as node 0, fully compatible with all
        compat = torch.ones((B, N + 1, N + 1), dtype=torch.bool, device=device)
        compat[:, 1:, 1:] = compat_parts

        # --- node_features (include SEP) ---
        # one-hot material (3)
        mat_oh = torch.nn.functional.one_hot(material, num_classes=self.material_types).float()  # [B,N,3]
        mot = motion.float().unsqueeze(-1)  # [B,N,1]
        siz = size.float().unsqueeze(-1)    # [B,N,1]
        part_feats = torch.cat([mat_oh, siz, mot], dim=-1)  # [B,N,5]
        sep_feats = torch.zeros((B, 1, self.node_feat_dim), device=device)
        node_features = torch.cat([sep_feats, part_feats], dim=1)  # [B,N+1,5]

        # --- also provide attributes aligned with node indices (SEP padded) ---
        material_all = torch.full((B, N + 1), -1, dtype=torch.long, device=device)
        motion_all = torch.full((B, N + 1), -1, dtype=torch.long, device=device)
        size_all = torch.zeros((B, N + 1), dtype=torch.float32, device=device)
        material_all[:, 1:] = material
        motion_all[:, 1:] = motion
        size_all[:, 1:] = size

        return TensorDict(
            {
                "node_features": node_features,
                "compat": compat,
                "material": material_all,
                "motion": motion_all,
                "size": size_all,
                "build_limit": torch.full((B, 1), self.build_limit, dtype=torch.float32, device=device),
            },
            batch_size=batch_size,
        )
