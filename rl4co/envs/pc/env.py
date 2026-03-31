from __future__ import annotations

import torch
from tensordict import TensorDict

from rl4co.envs.pc.generator import FPIGenerator


class RunningZScore:
    def __init__(self, eps: float = 1e-6):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.eps = eps

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        if self.count < 2:
            return value
        std = max(self.var, self.eps) ** 0.5
        return (value - self.mean) / std

    def update(self, value: torch.Tensor) -> None:
        val = float(value.detach().mean().item())
        self.count += 1
        if self.count == 1:
            self.mean = val
            self.var = 1.0
            return
        delta = val - self.mean
        self.mean += delta / self.count
        delta2 = val - self.mean
        # Welford-style running variance accumulator normalized by count.
        prev_m2 = self.var * max(self.count - 2, 1)
        m2 = prev_m2 + delta * delta2
        self.var = m2 / max(self.count - 1, 1)


class PartConsolidationEnv:
    """
    General-graph Part Consolidation environment.

    Action space:
        0     : SEP (close current open group)
        1..N  : choose one real part and add it to the current group

    Reward:
        terminal reward only
    """

    def __init__(
        self,
        generator: FPIGenerator | None = None,
        generator_params: dict | None = None,
        min_group_size_before_sep: int = 1,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.generator = generator or FPIGenerator(**(generator_params or {}))
        self.min_group_size_before_sep = int(min_group_size_before_sep)

        self.N = self.generator.num_nodes
        self.F = self.generator.node_feat_dim
        self._reward_static_td: TensorDict | None = None
        self._terminal_reward_stats = {
            "infeasible_solution": RunningZScore(),
            "num_groups": RunningZScore(),
            "total_internal_strength": RunningZScore(),
        }
        self._terminal_reward_weights = {
            "infeasible_solution": -200.0,
            "num_groups": -50.0,
            "total_internal_strength": 30.0,
        }

    def reset(self, batch_size: int) -> TensorDict:
        td = self.generator(batch_size=batch_size, device=self.device)
        B = batch_size

        assigned = torch.zeros((B, self.N), dtype=torch.bool, device=self.device)
        assigned[:, 0] = True

        open_group = torch.zeros((B, self.N), dtype=torch.bool, device=self.device)
        open_group_size = torch.zeros((B, 3), dtype=torch.float32, device=self.device)
        closed_group_count = torch.zeros((B,), dtype=torch.long, device=self.device)

        td_out = TensorDict(
            {
                **td,
                "assigned": assigned,
                "open_group": open_group,
                "open_group_size": open_group_size,
                "closed_group_count": closed_group_count,
                "fallback_part_mask": torch.zeros((B, self.N), dtype=torch.bool, device=self.device),
                "done": torch.zeros((B, 1), dtype=torch.bool, device=self.device),
                "action_mask": torch.ones((B, self.N), dtype=torch.bool, device=self.device),
            },
            batch_size=[B],
        )

        td_out["action_mask"] = self.get_action_mask(td_out)
        self._reward_static_td = td_out.clone()
        return td_out

    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        assigned = td["assigned"]
        open_group = td["open_group"]
        open_group_size = td["open_group_size"]
        size = td["size"]
        build_limit = td["build_limit"]
        assembly_adj = td["assembly_adj"]
        compat = td["compat"]

        B, N = assigned.shape
        mask = torch.zeros((B, N), dtype=torch.bool, device=assigned.device)
        fallback_part_mask = torch.zeros((B, N), dtype=torch.bool, device=assigned.device)

        open_any = open_group.any(dim=-1)
        all_assigned = assigned[:, 1:].all(dim=-1)

        # --------------------------
        # Case 1: open group exists
        # --------------------------
        if open_any.any():
            connected_to_group = (assembly_adj & open_group.unsqueeze(1)).any(dim=-1)

            direct_open_neighbors = assembly_adj & open_group.unsqueeze(1)
            compat_ok = (~direct_open_neighbors | compat).all(dim=-1)

            projected_group_size = open_group_size.unsqueeze(1) + size
            volume_ok = (projected_group_size <= build_limit.unsqueeze(1)).all(dim=-1)

            feasible_part = (~assigned) & connected_to_group & compat_ok & volume_ok
            feasible_part[:, 0] = False

            # fallback
            no_adjacent_feasible = feasible_part[:, 1:].sum(dim=-1) == 0
            if no_adjacent_feasible.any():
                rows = torch.where(no_adjacent_feasible)[0]
                fallback_part_mask[rows] = (~assigned[rows]) & volume_ok[rows]
                fallback_part_mask[rows, 0] = False
                feasible_part[rows] = fallback_part_mask[rows]

            mask = feasible_part

            group_card = open_group[:, 1:].sum(dim=-1)
            sep_allowed = open_any & (group_card >= self.min_group_size_before_sep)
            sep_allowed = sep_allowed | (open_any & (~mask[:, 1:].any(dim=-1)))
            mask[:, 0] = sep_allowed

        # --------------------------
        # Case 2: no open group
        # --------------------------
        no_open_rows = ~open_any
        if no_open_rows.any():
            rows = torch.where(no_open_rows)[0]
            mask[rows] = ~assigned[rows]
            mask[rows, 0] = False

        # --------------------------
        # finished cases
        # --------------------------
        finished_but_open = all_assigned & open_any
        if finished_but_open.any():
            rows = torch.where(finished_but_open)[0]
            mask[rows] = False
            mask[rows, 0] = True

        finished_and_closed = all_assigned & (~open_any)
        if finished_and_closed.any():
            rows = torch.where(finished_and_closed)[0]
            mask[rows] = False
            mask[rows, 0] = True

        td["fallback_part_mask"] = fallback_part_mask
        return mask

    def step(self, td: TensorDict, action: torch.Tensor) -> TensorDict:
        B = td.batch_size[0]
        action = action.long().view(B)

        assigned = td["assigned"].clone()
        open_group = td["open_group"].clone()
        open_group_size = td["open_group_size"].clone()
        closed_group_count = td["closed_group_count"].clone()
        size = td["size"]

        is_sep = action.eq(0)
        is_part = ~is_sep

        # SEP
        if is_sep.any():
            rows = torch.where(is_sep)[0]
            had_open = open_group[rows].any(dim=-1)
            closed_group_count[rows] += had_open.long()
            open_group[rows] = False
            open_group_size[rows] = 0.0

        # PART
        if is_part.any():
            rows = torch.where(is_part)[0]
            idx = action[is_part]
            assigned[rows, idx] = True
            open_group[rows, idx] = True
            open_group_size[rows] += size[rows, idx, :]

        all_assigned = assigned[:, 1:].all(dim=-1)
        open_empty = ~open_group.any(dim=-1)
        done = (all_assigned & open_empty).view(B, 1)

        td2 = td.clone()
        td2.update(
            {
                "assigned": assigned,
                "open_group": open_group,
                "open_group_size": open_group_size,
                "closed_group_count": closed_group_count,
                "done": done,
            }
        )

        td2["action_mask"] = self.get_action_mask(td2)
        return td2

    def reward_from_actions(self, actions: torch.Tensor) -> torch.Tensor:
        groups = self.actions_to_groups(actions, N=self.N)
        raw = self._terminal_reward_components(groups, device=actions.device)
        reward = torch.zeros_like(raw["num_groups"])
        for name, values in raw.items():
            reward = reward + self._terminal_reward_weights[name] * self._terminal_reward_stats[name].normalize(values)
        for name, values in raw.items():
            self._terminal_reward_stats[name].update(values)
        return reward

    @staticmethod
    def actions_to_groups(actions: torch.Tensor, N: int) -> list[list[list[int]]]:
        B, T = actions.shape
        out = []

        for b in range(B):
            groups_b = []
            cur = []
            used = set()

            for t in range(T):
                a = int(actions[b, t].item())
                if a == 0:
                    if cur:
                        groups_b.append(cur)
                        cur = []
                else:
                    if 0 < a < N and a not in used:
                        cur.append(a)
                        used.add(a)

            if cur:
                groups_b.append(cur)
            out.append(groups_b)

        return out

    def _terminal_reward_components(self, groups: list[list[list[int]]], device: torch.device) -> dict[str, torch.Tensor]:
        if self._reward_static_td is None:
            raise RuntimeError("reward_from_actions called before env.reset")

        td = self._reward_static_td
        B = len(groups)
        infeasible_solution = torch.zeros((B,), dtype=torch.float32, device=device)
        num_groups = torch.tensor([len(g) for g in groups], dtype=torch.float32, device=device)
        total_internal_strength = torch.zeros((B,), dtype=torch.float32, device=device)

        compat = td["compat"]
        size = td["size"]
        build_limit = td["build_limit"]
        isstandard = td["isstandard"]

        for b, groups_b in enumerate(groups):
            infeasible = False
            for group in groups_b:
                total_internal_strength[b] += self._group_internal_strength(group, td["W"][b])
                if not self._group_feasible(group, compat[b], size[b], build_limit[b], isstandard[b], td["assembly_adj"][b]):
                    infeasible = True
            infeasible_solution[b] = float(infeasible)

        return {
            "infeasible_solution": infeasible_solution,
            "num_groups": num_groups,
            "total_internal_strength": total_internal_strength,
        }

    def _group_internal_strength(self, group: list[int], w: torch.Tensor) -> torch.Tensor:
        total = torch.tensor(0.0, device=w.device)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                total = total + w[group[i], group[j]]
        return total

    def _group_feasible(
        self,
        group: list[int],
        compat: torch.Tensor,
        size: torch.Tensor,
        build_limit: torch.Tensor,
        isstandard: torch.Tensor,
        assembly_adj: torch.Tensor,
    ) -> bool:
        if not group:
            return True
        if isstandard[group].bool().any():
            return False
        if not torch.all(size[group].sum(dim=0) <= build_limit):
            return False
        for i in group:
            for j in group:
                if not bool(compat[i, j].item()):
                    return False
        visited = {group[0]}
        stack = [group[0]]
        while stack:
            cur = stack.pop()
            for nxt in group:
                if bool(assembly_adj[cur, nxt].item()) and nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        return len(visited) == len(group)
