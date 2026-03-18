from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict import TensorDict


@dataclass
class FPIGeneratorParams:
    # number of real parts (excluding SEP)
    num_parts: int = 13

    # node attribute ranges
    material_types: int = 3
    L_low: float = 5.0
    L_high: float = 220.0
    W_low: float = 5.0
    W_high: float = 80.0
    H_low: float = 0.5
    H_high: float = 30.0

    # node attribute probabilities
    p_maint_H: float = 0.10
    p_standard: float = 0.08

    # interaction graph density
    p_edge: float = 0.35

    # edge attribute probability
    p_relative_motion: float = 0.15

    # build limits for feasibility
    build_limit_L: float = 250.0
    build_limit_W: float = 120.0
    build_limit_H: float = 80.0


class FPIGenerator:
    """
    FPI network generator.

    Important design:
    1) FPI network = interaction graph (policy input)
    2) compat = feasibility constraint base (action mask basis)
    3) W is NOT filtered by compat
    4) group-level build feasibility is checked in env.get_action_mask()
    """

    def __init__(self, **kwargs):
        self.p = FPIGeneratorParams(**kwargs)
        self.num_parts = self.p.num_parts
        self.num_nodes = self.num_parts + 1  # + SEP
        self.node_feat_dim = self.p.material_types + 3 + 1 + 1
        self.edge_feat_dim = 1 + 3 + 1 + 1  # mat_var + stack_size(3) + maint_diff + rel_motion

        self.build_limit = torch.tensor(
            [self.p.build_limit_L, self.p.build_limit_W, self.p.build_limit_H],
            dtype=torch.float32,
        )

    def __call__(self, batch_size: int, device: torch.device | str = "cpu") -> TensorDict:
        device = torch.device(device)
        B = batch_size
        N = self.num_parts
        eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)

        # 1) node attributes
        material = torch.randint(0, self.p.material_types, (B, N), device=device)

        L = torch.rand((B, N), device=device) * (self.p.L_high - self.p.L_low) + self.p.L_low
        Wd = torch.rand((B, N), device=device) * (self.p.W_high - self.p.W_low) + self.p.W_low
        H = torch.rand((B, N), device=device) * (self.p.H_high - self.p.H_low) + self.p.H_low
        size = torch.stack([L, Wd, H], dim=-1).float()

        maintfreq = (torch.rand((B, N), device=device) < self.p.p_maint_H).long()
        isstandard = (torch.rand((B, N), device=device) < self.p.p_standard).long()

        # 2) interaction graph W
        edge_exist = (torch.rand((B, N, N), device=device) < self.p.p_edge)
        edge_exist = torch.triu(edge_exist, diagonal=1)
        edge_exist = edge_exist | edge_exist.transpose(-1, -2)
        edge_exist = edge_exist & (~eye)

        raw_w = 0.5 + 0.5 * torch.rand((B, N, N), device=device)
        W_parts = edge_exist.float() * raw_w
        W_parts = 0.5 * (W_parts + W_parts.transpose(-1, -2))
        W_parts = W_parts * (~eye).float()

        # 3) edge attributes
        mat_var = (material.unsqueeze(-1) != material.unsqueeze(-2)).long()
        stack_size = size.unsqueeze(-2) + size.unsqueeze(-3)
        maint_diff = (maintfreq.unsqueeze(-1) != maintfreq.unsqueeze(-2)).long()

        rel = (torch.rand((B, N, N), device=device) < self.p.p_relative_motion)
        rel = torch.triu(rel, diagonal=1)
        rel = rel | rel.transpose(-1, -2)
        rel = rel & (~eye)
        rel_motion = rel.long()

        build = self.build_limit.to(device).view(1, 1, 1, 3)
        stack_ok = (stack_size <= build).all(dim=-1)

        # 4) feasibility matrix (pairwise compatibility base)
        compat_parts = (
            (mat_var == 0)
            & (maint_diff == 0)
            & (rel_motion == 0)
            & stack_ok
        )

        # standard part cannot be merged with others
        standard_pair_block = isstandard.unsqueeze(-1) | isstandard.unsqueeze(-2)
        compat_parts = compat_parts & (~standard_pair_block)
        compat_parts = compat_parts | eye

        # 5) node features
        mat_oh = torch.nn.functional.one_hot(material, num_classes=self.p.material_types).float()
        part_node_features = torch.cat(
            [
                mat_oh,
                size.float(),
                maintfreq.float().unsqueeze(-1),
                isstandard.float().unsqueeze(-1),
            ],
            dim=-1,
        )

        # 6) edge features = raw stack_size
        part_edge_features = torch.cat(
            [
                mat_var.float().unsqueeze(-1),
                stack_size.float(),
                maint_diff.float().unsqueeze(-1),
                rel_motion.float().unsqueeze(-1),
            ],
            dim=-1,
        )

        # 7) pad SEP node (index 0)
        material_all = torch.full((B, N + 1), -1, dtype=torch.long, device=device)
        maint_all = torch.full((B, N + 1), -1, dtype=torch.long, device=device)
        std_all = torch.full((B, N + 1), -1, dtype=torch.long, device=device)
        size_all = torch.zeros((B, N + 1, 3), dtype=torch.float32, device=device)

        material_all[:, 1:] = material
        maint_all[:, 1:] = maintfreq
        std_all[:, 1:] = isstandard
        size_all[:, 1:] = size

        node_features = torch.zeros((B, N + 1, self.node_feat_dim), dtype=torch.float32, device=device)
        node_features[:, 1:, :] = part_node_features

        W = torch.zeros((B, N + 1, N + 1), dtype=torch.float32, device=device)
        W[:, 1:, 1:] = W_parts

        mat_var_all = torch.zeros((B, N + 1, N + 1), dtype=torch.float32, device=device)
        maint_diff_all = torch.zeros((B, N + 1, N + 1), dtype=torch.float32, device=device)
        rel_motion_all = torch.zeros((B, N + 1, N + 1), dtype=torch.float32, device=device)
        stack_all = torch.zeros((B, N + 1, N + 1, 3), dtype=torch.float32, device=device)
        edge_features = torch.zeros((B, N + 1, N + 1, self.edge_feat_dim), dtype=torch.float32, device=device)

        mat_var_all[:, 1:, 1:] = mat_var.float()
        maint_diff_all[:, 1:, 1:] = maint_diff.float()
        rel_motion_all[:, 1:, 1:] = rel_motion.float()
        stack_all[:, 1:, 1:, :] = stack_size
        edge_features[:, 1:, 1:, :] = part_edge_features

        compat = torch.ones((B, N + 1, N + 1), dtype=torch.bool, device=device)
        compat[:, 1:, 1:] = compat_parts

        build_limit = self.build_limit.to(device).unsqueeze(0).repeat(B, 1)

        return TensorDict(
            {
                "node_features": node_features,
                "edge_features": edge_features,
                "material": material_all,
                "size": size_all,
                "maintfreq": maint_all,
                "isstandard": std_all,
                "W": W,
                "mat_var": mat_var_all,
                "stack_size": stack_all,
                "maint_diff": maint_diff_all,
                "rel_motion": rel_motion_all,
                "compat": compat,
                "build_limit": build_limit,
            },
            batch_size=[B],
        )