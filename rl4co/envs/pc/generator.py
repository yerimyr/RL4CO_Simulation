from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from tensordict import TensorDict


@dataclass
class FPIGeneratorParams:
    # number of real parts (excluding SEP)
    num_parts: int = 4

    # node attribute ranges
    material_types: int = 3
    L_low: float = 5.0
    L_high: float = 160.0
    W_low: float = 5.0
    W_high: float = 70.0
    H_low: float = 0.5
    H_high: float = 30.0

    # node attribute probabilities
    p_maint_H: float = 0.20
    p_standard: float = 0.10

    # edge attribute probability
    p_relative_motion: float = 0.15
    p_extra_edge: float = 0.30

    # topology generation
    topology_mode: str = "mixed"
    p_chain: float = 0.15
    p_star: float = 0.15
    p_tree: float = 0.20
    p_two_module_bridge: float = 0.20
    p_dense_clustered: float = 0.15
    p_sparse_random: float = 0.15

    # build limits
    build_limit_L: float = 360.0
    build_limit_W: float = 180.0
    build_limit_H: float = 120.0


class FPIGenerator:
    """
    General-graph FPI generator.

    It creates:
    1) node attributes
       - material
       - size (L, W, H)
       - maintfreq
       - isstandard
    2) pairwise relation tensors
       - material variance
       - stack size
       - maintfreq diff
       - relative motion
    3) positional adjacency matrix (assembly_adj)
       - True if two parts are physically attached
       - False otherwise

    Important consistency rule:
    - If assembly_adj[i, j] == False, then pair relation attributes are forced to 0.
    - Therefore non-attached pairs never carry valid physical relation data.
    """

    def __init__(self, **kwargs):
        self.p = FPIGeneratorParams(**kwargs)
        self.num_parts = self.p.num_parts
        self.num_nodes = self.num_parts + 1  # + SEP token
        self.topology_names = [
            "chain",
            "star",
            "tree",
            "two_module_bridge",
            "dense_clustered",
            "sparse_random",
        ]

        # material one-hot + size(3) + maintfreq + isstandard + normalized degree
        self.node_feat_dim = self.p.material_types + 3 + 1 + 1 + 1

        # adjacency + material variance + stack size(3) + maint diff + relative motion
        self.edge_feat_dim = 1 + 1 + 3 + 1 + 1

        self.build_limit = torch.tensor(
            [self.p.build_limit_L, self.p.build_limit_W, self.p.build_limit_H],
            dtype=torch.float32,
        )

    def _add_undirected_edge(self, adj: torch.Tensor, i: int, j: int) -> None:
        if i == j:
            return
        adj[i, j] = True
        adj[j, i] = True

    def _connect_sequence(self, adj: torch.Tensor, nodes: list[int]) -> None:
        for idx in range(len(nodes) - 1):
            self._add_undirected_edge(adj, nodes[idx], nodes[idx + 1])

    def _sample_topology_id(self, device: torch.device) -> int:
        mode = self.p.topology_mode
        if mode != "mixed":
            if mode not in self.topology_names:
                raise ValueError(f"Unknown topology_mode: {mode}")
            return self.topology_names.index(mode)

        probs = torch.tensor(
            [
                self.p.p_chain,
                self.p.p_star,
                self.p.p_tree,
                self.p.p_two_module_bridge,
                self.p.p_dense_clustered,
                self.p.p_sparse_random,
            ],
            dtype=torch.float32,
            device=device,
        )
        probs = probs / probs.sum().clamp_min(1e-8)
        return int(torch.multinomial(probs, num_samples=1).item())

    def _build_chain_adjacency(self, device: torch.device) -> torch.Tensor:
        n = self.num_parts
        adj = torch.zeros((n, n), dtype=torch.bool, device=device)
        order = torch.randperm(n, device=device).tolist()
        self._connect_sequence(adj, order)
        return adj

    def _build_star_adjacency(self, device: torch.device) -> torch.Tensor:
        n = self.num_parts
        adj = torch.zeros((n, n), dtype=torch.bool, device=device)
        center = int(torch.randint(0, n, (1,), device=device).item())
        for node in range(n):
            if node != center:
                self._add_undirected_edge(adj, center, node)
        return adj

    def _build_tree_adjacency(self, device: torch.device) -> torch.Tensor:
        n = self.num_parts
        adj = torch.zeros((n, n), dtype=torch.bool, device=device)
        nodes = torch.randperm(n, device=device)
        for i in range(1, n):
            child = int(nodes[i].item())
            parent_idx = int(torch.randint(0, i, (1,), device=device).item())
            parent = int(nodes[parent_idx].item())
            self._add_undirected_edge(adj, parent, child)
        return adj

    def _build_two_module_bridge_adjacency(self, device: torch.device) -> torch.Tensor:
        n = self.num_parts
        adj = torch.zeros((n, n), dtype=torch.bool, device=device)
        order = torch.randperm(n, device=device).tolist()
        split = max(1, n // 2)
        if split >= n:
            split = n - 1
        left = order[:split]
        right = order[split:]

        self._connect_sequence(adj, left)
        self._connect_sequence(adj, right)

        for group in (left, right):
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    if not adj[group[i], group[j]] and torch.rand(1, device=device).item() < 0.65:
                        self._add_undirected_edge(adj, group[i], group[j])

        self._add_undirected_edge(adj, left[-1], right[0])
        return adj

    def _build_dense_clustered_adjacency(self, device: torch.device) -> torch.Tensor:
        n = self.num_parts
        adj = torch.zeros((n, n), dtype=torch.bool, device=device)
        order = torch.randperm(n, device=device).tolist()
        num_clusters = 3 if n >= 6 else 2
        splits = np.array_split(order, num_clusters)
        clusters = [list(map(int, split.tolist() if hasattr(split, "tolist") else split)) for split in splits if len(split) > 0]

        for cluster in clusters:
            self._connect_sequence(adj, cluster)
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    if not adj[cluster[i], cluster[j]] and torch.rand(1, device=device).item() < 0.85:
                        self._add_undirected_edge(adj, cluster[i], cluster[j])

        for idx in range(len(clusters) - 1):
            self._add_undirected_edge(adj, clusters[idx][-1], clusters[idx + 1][0])
            if torch.rand(1, device=device).item() < 0.20:
                self._add_undirected_edge(adj, clusters[idx][0], clusters[idx + 1][-1])
        return adj

    def _build_sparse_random_connected_adjacency(self, device: torch.device) -> torch.Tensor:
        adj = self._build_tree_adjacency(device)
        n = self.num_parts
        extra_prob = min(self.p.p_extra_edge, 0.12)
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i, j] and torch.rand(1, device=device).item() < extra_prob:
                    self._add_undirected_edge(adj, i, j)
        return adj

    def _build_adjacency_from_topology(self, topology_id: int, device: torch.device) -> torch.Tensor:
        if topology_id == 0:
            return self._build_chain_adjacency(device)
        if topology_id == 1:
            return self._build_star_adjacency(device)
        if topology_id == 2:
            return self._build_tree_adjacency(device)
        if topology_id == 3:
            return self._build_two_module_bridge_adjacency(device)
        if topology_id == 4:
            return self._build_dense_clustered_adjacency(device)
        if topology_id == 5:
            return self._build_sparse_random_connected_adjacency(device)
        raise ValueError(f"Unknown topology id: {topology_id}")

    def _build_general_graph_adjacency(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create a CONNECTED undirected graph for each batch instance from
        a mixture of topology families. This gives the random generator
        structural diversity, not just attribute diversity.
        """
        B = batch_size
        N = self.num_parts

        adj = torch.zeros((B, N, N), dtype=torch.bool, device=device)
        topology_id = torch.zeros((B,), dtype=torch.long, device=device)

        for b in range(B):
            topo_id = self._sample_topology_id(device)
            topology_id[b] = topo_id
            adj[b] = self._build_adjacency_from_topology(topo_id, device)

        degree = adj.sum(dim=-1).float()
        max_degree = degree.max(dim=-1, keepdim=True).values.clamp_min(1.0)
        pos1d = (degree / max_degree).unsqueeze(-1)
        return adj, pos1d, topology_id

    def __call__(self, batch_size: int, device: torch.device | str = "cpu") -> TensorDict:
        device = torch.device(device)
        B = batch_size
        N = self.num_parts
        eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)

        # ------------------------------
        # 1) node attributes
        # ------------------------------
        material = torch.randint(0, self.p.material_types, (B, N), device=device)

        L = torch.rand((B, N), device=device) * (self.p.L_high - self.p.L_low) + self.p.L_low
        Wd = torch.rand((B, N), device=device) * (self.p.W_high - self.p.W_low) + self.p.W_low
        H = torch.rand((B, N), device=device) * (self.p.H_high - self.p.H_low) + self.p.H_low
        size = torch.stack([L, Wd, H], dim=-1).float()

        maintfreq = (torch.rand((B, N), device=device) < self.p.p_maint_H).long()
        isstandard = (torch.rand((B, N), device=device) < self.p.p_standard).long()

        # ------------------------------
        # 2) general graph adjacency
        # ------------------------------
        adj_parts, pos1d, topology_id = self._build_general_graph_adjacency(B, device=device)

        # ------------------------------
        # 3) pair relations (ONLY for adjacent pairs)
        # ------------------------------
        mat_var_full = material.unsqueeze(-1) != material.unsqueeze(-2)
        stack_size_full = size.unsqueeze(-2) + size.unsqueeze(-3)
        maint_diff_full = maintfreq.unsqueeze(-1) != maintfreq.unsqueeze(-2)

        rel = torch.rand((B, N, N), device=device) < self.p.p_relative_motion
        rel = torch.triu(rel, diagonal=1)
        rel = rel | rel.transpose(-1, -2)
        rel = rel & (~eye)

        # consistency: relation values only exist on physically attached pairs
        mat_var = mat_var_full & adj_parts
        stack_size = stack_size_full * adj_parts.unsqueeze(-1).float()
        maint_diff = maint_diff_full & adj_parts
        rel_motion = rel & adj_parts

        build = self.build_limit.to(device).view(1, 1, 1, 3)
        stack_ok = (stack_size_full <= build).all(dim=-1)

        # basic direct-pair compatibility for adjacent pairs only
        compat_parts = (
            adj_parts
            & (~mat_var)
            & (~maint_diff)
            & (~rel_motion)
            & stack_ok
        )

        standard_pair_block = isstandard.unsqueeze(-1).bool() | isstandard.unsqueeze(-2).bool()
        compat_parts = compat_parts & (~standard_pair_block)
        compat_parts = compat_parts | eye

        # ------------------------------
        # 4) node features
        # ------------------------------
        mat_oh = torch.nn.functional.one_hot(material, num_classes=self.p.material_types).float()
        part_node_features = torch.cat(
            [
                mat_oh,
                size,
                maintfreq.float().unsqueeze(-1),
                isstandard.float().unsqueeze(-1),
                pos1d.float(),
            ],
            dim=-1,
        )

        # ------------------------------
        # 5) edge features
        # ------------------------------
        part_edge_features = torch.cat(
            [
                adj_parts.float().unsqueeze(-1),
                mat_var.float().unsqueeze(-1),
                stack_size.float(),
                maint_diff.float().unsqueeze(-1),
                rel_motion.float().unsqueeze(-1),
            ],
            dim=-1,
        )

        # ------------------------------
        # 6) pad SEP token at index 0
        # ------------------------------
        material_all = torch.full((B, N + 1), -1, dtype=torch.long, device=device)
        maint_all = torch.full((B, N + 1), -1, dtype=torch.long, device=device)
        std_all = torch.full((B, N + 1), -1, dtype=torch.long, device=device)
        size_all = torch.zeros((B, N + 1, 3), dtype=torch.float32, device=device)
        pos_all = torch.zeros((B, N + 1, 1), dtype=torch.float32, device=device)

        material_all[:, 1:] = material
        maint_all[:, 1:] = maintfreq
        std_all[:, 1:] = isstandard
        size_all[:, 1:] = size
        pos_all[:, 1:, :] = pos1d

        node_features = torch.zeros((B, N + 1, self.node_feat_dim), dtype=torch.float32, device=device)
        node_features[:, 1:, :] = part_node_features

        # ============================================================
        # 🔥 WEIGHTED W (수정된 부분)
        # ============================================================

        alpha = 0.4   # material 영향
        beta = 0.3    # maintfreq 영향
        gamma = 0.8   # rel_motion 강하게

        score = (
            1.0
            - alpha * mat_var.float()
            - beta * maint_diff.float()
            - gamma * rel_motion.float()
        )

        score = torch.clamp(score, 0.1, 1.0)

        # adjacency=1인 경우만 weight 부여
        W_parts = adj_parts.float() * score

        # 기존 구조 유지 (SEP 포함 padding)
        W = torch.zeros((B, N + 1, N + 1), dtype=torch.float32, device=device)
        W[:, 1:, 1:] = W_parts

        assembly_adj = torch.zeros((B, N + 1, N + 1), dtype=torch.bool, device=device)
        assembly_adj[:, 1:, 1:] = adj_parts

        mat_var_all = torch.zeros((B, N + 1, N + 1), dtype=torch.float32, device=device)
        maint_diff_all = torch.zeros((B, N + 1, N + 1), dtype=torch.float32, device=device)
        rel_motion_all = torch.zeros((B, N + 1, N + 1), dtype=torch.float32, device=device)
        stack_all = torch.zeros((B, N + 1, N + 1, 3), dtype=torch.float32, device=device)
        edge_features = torch.zeros((B, N + 1, N + 1, self.edge_feat_dim), dtype=torch.float32, device=device)

        mat_var_all[:, 1:, 1:] = mat_var.float()
        maint_diff_all[:, 1:, 1:] = maint_diff.float()
        rel_motion_all[:, 1:, 1:] = rel_motion.float()
        stack_all[:, 1:, 1:, :] = stack_size.float()
        edge_features[:, 1:, 1:, :] = part_edge_features

        compat = torch.ones((B, N + 1, N + 1), dtype=torch.bool, device=device)
        compat[:, 1:, 1:] = compat_parts

        relation_valid = torch.zeros((B, N + 1, N + 1), dtype=torch.bool, device=device)
        relation_valid[:, 1:, 1:] = adj_parts
        relation_consistent = torch.ones((B,), dtype=torch.bool, device=device)

        build_limit = self.build_limit.to(device).unsqueeze(0).repeat(B, 1)

        return TensorDict(
            {
                "node_features": node_features,
                "edge_features": edge_features,
                "topology_id": topology_id,
                "material": material_all,
                "size": size_all,
                "maintfreq": maint_all,
                "isstandard": std_all,
                "pos1d": pos_all,
                "W": W,
                "assembly_adj": assembly_adj,
                "mat_var": mat_var_all,
                "stack_size": stack_all,
                "maint_diff": maint_diff_all,
                "rel_motion": rel_motion_all,
                "compat": compat,
                "relation_valid": relation_valid,
                "relation_consistent": relation_consistent,
                "build_limit": build_limit,
            },
            batch_size=[B],
        )
