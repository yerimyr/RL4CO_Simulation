from __future__ import annotations

import torch
from tensordict import TensorDict

from rl4co.envs.pc.generator import FPIGenerator


class PartConsolidationEnv:
    def __init__(
        self,
        generator: FPIGenerator | None = None,
        generator_params: dict | None = None,
        reward_a: float = 1.0,
        reward_b: float = 0.2,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.generator = generator or FPIGenerator(**(generator_params or {}))
        self.reward_a = float(reward_a)
        self.reward_b = float(reward_b)

        self.N = self.generator.num_nodes
        self.F = self.generator.node_feat_dim

    def reset(self, batch_size: int) -> TensorDict:
        td = self.generator(batch_size=batch_size, device=self.device)

        assigned = torch.zeros((batch_size, self.N), dtype=torch.bool, device=self.device)
        assigned[:, 0] = True  # SEP

        open_group = torch.zeros((batch_size, self.N), dtype=torch.bool, device=self.device)
        open_group_size = torch.zeros((batch_size, 3), dtype=torch.float32, device=self.device)

        out = TensorDict(
            {
                **td,
                "assigned": assigned,
                "open_group": open_group,
                "open_group_size": open_group_size,
                "done": torch.zeros((batch_size, 1), dtype=torch.bool, device=self.device),
                "action_mask": torch.ones((batch_size, self.N), dtype=torch.bool, device=self.device),
            },
            batch_size=[batch_size],
        )
        out["action_mask"] = self.get_action_mask(out)
        return out

    def step(self, td: TensorDict, action: torch.Tensor) -> TensorDict:
        B = td.batch_size[0]
        action = action.long().view(B)

        assigned = td["assigned"].clone()
        open_group = td["open_group"].clone()
        open_group_size = td["open_group_size"].clone()
        size = td["size"]

        is_sep = action.eq(0)
        is_part = ~is_sep

        # close current group
        if is_sep.any():
            open_group[is_sep] = False
            open_group_size[is_sep] = 0.0

        # add selected part into current group
        if is_part.any():
            rows = torch.arange(B, device=td.device)[is_part]
            idx = action[is_part]
            assigned[rows, idx] = True
            open_group[rows, idx] = True
            open_group_size[rows] = open_group_size[rows] + size[rows, idx, :]

        all_assigned = assigned[:, 1:].all(dim=-1)
        open_empty = ~open_group.any(dim=-1)
        done = (all_assigned & open_empty).view(B, 1)

        td2 = td.clone()
        td2.update(
            {
                "assigned": assigned,
                "open_group": open_group,
                "open_group_size": open_group_size,
                "done": done,
            }
        )
        td2["action_mask"] = self.get_action_mask(td2)
        return td2

    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        compat = td["compat"]
        assigned = td["assigned"]
        open_group = td["open_group"]
        size = td["size"]
        build_limit = td["build_limit"]
        open_group_size = td["open_group_size"]

        B, N = assigned.shape
        mask = ~assigned

        # pairwise compatibility with all members currently in the open group
        open_any = open_group.any(dim=-1)
        if open_any.any():
            allowed = torch.ones((B, N), dtype=torch.bool, device=td.device)
            for j in range(N):
                sel = open_group[:, j]
                if sel.any():
                    allowed = torch.where(sel.unsqueeze(-1), allowed & compat[:, j, :], allowed)
            mask = mask & allowed

        # group-level build feasibility:
        # current open group total size + candidate part size <= build limit
        projected_group_size = open_group_size.unsqueeze(1) + size
        volume_ok = (projected_group_size <= build_limit.unsqueeze(1)).all(dim=-1)
        mask = mask & volume_ok

        all_assigned = assigned[:, 1:].all(dim=-1)
        sep_allowed = open_any | all_assigned
        mask[:, 0] = sep_allowed

        no_feasible = mask.sum(dim=-1) == 0
        if no_feasible.any():
            mask[no_feasible] = False
            mask[no_feasible, 0] = True

        return mask

    def reward_from_actions(self, W: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        groups = self.actions_to_groups(actions, N=W.size(-1))
        K = torch.tensor([len(g) for g in groups], dtype=torch.float32, device=W.device)
        Q = self.batch_modularity(W, groups)
        return self.reward_a * Q - self.reward_b * K

    @staticmethod
    def actions_to_groups(actions: torch.Tensor, N: int) -> list[list[list[int]]]:
        B, T = actions.shape
        out: list[list[list[int]]] = []
        for b in range(B):
            gs: list[list[int]] = []
            cur: list[int] = []
            for t in range(T):
                a = int(actions[b, t].item())
                if a == 0:
                    if len(cur) > 0:
                        gs.append(cur)
                        cur = []
                else:
                    if 0 < a < N and a not in cur:
                        cur.append(a)
            if len(cur) > 0:
                gs.append(cur)
            out.append(gs)
        return out

    @staticmethod
    def batch_modularity(W: torch.Tensor, groups: list[list[list[int]]]) -> torch.Tensor:
        """
        Modularity based on the document formula:
            Q = (1 / 2m) * sum_ij [ A_ij - (s_i s_j / 2m) ] * delta(c_i, c_j)

        Here:
        - A_ij: binary adjacency derived from W (interaction exists or not)
        - s_i: degree of node i
        - m: number of edges
        - SEP node (index 0) is excluded from the calculation
        """
        B = W.shape[0]
        Qs = torch.zeros((B,), dtype=torch.float32, device=W.device)

        for b in range(B):
            A = (W[b, 1:, 1:] > 0).float()
            A = torch.triu(A, diagonal=1)
            A = A + A.t()

            two_m = A.sum()
            if two_m.item() <= 1e-8:
                Qs[b] = 0.0
                continue

            degree = A.sum(dim=1)
            num_parts = A.size(0)
            labels = torch.arange(num_parts, device=W.device)

            for ci, members in enumerate(groups[b]):
                for node in members:
                    if node > 0:
                        labels[node - 1] = ci

            same = labels.view(-1, 1).eq(labels.view(1, -1)).float()
            expected = torch.outer(degree, degree) / two_m
            Q = ((A - expected) * same).sum() / two_m
            Qs[b] = Q

        return Qs