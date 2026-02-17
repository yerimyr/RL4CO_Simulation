from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import Tensor
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.pylogger import get_pylogger

from .generator import PCGenerator

log = get_pylogger(__name__)


class PartConsolidationEnv(RL4COEnvBase):

    name = "pc"

    def __init__(
        self,
        generator: PCGenerator | None = None,
        generator_params: dict = {},
        w_material: float = 30.0,
        w_motion: float = 30.0,
        w_size: float = 60.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if generator is None:
            generator = PCGenerator(**generator_params)

        self.generator = generator
        self.w_material = float(w_material)
        self.w_motion = float(w_motion)
        self.w_size = float(w_size)

        self._make_spec(self.generator)

    # =====================================================
    # RESET
    # =====================================================
    def _reset(self, td: TensorDict | None = None, batch_size=None) -> TensorDict:

        device = td.device
        N = td["node_features"].shape[-2]  # includes SEP

        current_node = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        assigned = torch.zeros((*batch_size, N), dtype=torch.bool, device=device)
        assigned[..., 0] = True  # SEP already assigned

        open_group = torch.zeros((*batch_size, N), dtype=torch.bool, device=device)
        group_count = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)

        out = TensorDict(
            {
                "node_features": td["node_features"],
                "compat": td["compat"],
                "material": td["material"],
                "motion": td["motion"],
                "size": td["size"],
                "build_limit": td["build_limit"],
                "current_node": current_node,
                "i": i,
                "assigned": assigned,
                "open_group": open_group,
                "group_count": group_count,
                "action_mask": torch.ones((*batch_size, N), dtype=torch.bool, device=device),
                "reward": torch.zeros((*batch_size, 1), dtype=torch.float32, device=device),
                "done": torch.zeros((*batch_size, 1), dtype=torch.bool, device=device),
            },
            batch_size=batch_size,
        )

        out["action_mask"] = self.get_action_mask(out)
        return out

    # =====================================================
    # STEP
    # =====================================================
    def _step(self, td: TensorDict) -> TensorDict:

        action = td["action"].long().view(*td.batch_size, 1)

        assigned = td["assigned"].clone()
        open_group = td["open_group"].clone()
        group_count = td["group_count"].clone()

        is_sep = action.eq(0)
        has_open = open_group.any(dim=-1, keepdim=True)

        # Close group if SEP chosen and group not empty
        close_group = is_sep & has_open
        group_count = group_count + close_group.long()

        # Clear open group when SEP chosen
        open_group = torch.where(
            is_sep.expand_as(open_group),
            torch.zeros_like(open_group),
            open_group,
        )

        # ------------------------------
        # FIXED scatter logic (중요)
        # ------------------------------
        is_part = ~is_sep

        if is_part.any():
            idx = action.squeeze(-1)

            assigned_sc = assigned.scatter(-1, idx.unsqueeze(-1), True)
            open_group_sc = open_group.scatter(-1, idx.unsqueeze(-1), True)

            assigned = torch.where(
                is_part.expand_as(assigned), assigned_sc, assigned
            )
            open_group = torch.where(
                is_part.expand_as(open_group), open_group_sc, open_group
            )

        all_assigned = assigned[..., 1:].all(dim=-1, keepdim=True)
        open_empty = ~open_group.any(dim=-1, keepdim=True)
        done = all_assigned & open_empty

        td.update(
            {
                "current_node": action,
                "i": td["i"] + 1,
                "assigned": assigned,
                "open_group": open_group,
                "group_count": group_count,
                "reward": torch.zeros((*td.batch_size, 1), device=td.device),  # 최종 reward는 get_reward()에서 계산
                "done": done,
            }
        )

        td["action_mask"] = self.get_action_mask(td)
        return td

    # =====================================================
    # SPEC
    # =====================================================
    def _make_spec(self, generator: PCGenerator):
        N = generator.num_nodes
        F = generator.node_feat_dim

        self.observation_spec = Composite(
            node_features=Unbounded(shape=(N, F)),
            compat=Unbounded(shape=(N, N)),
            material=Unbounded(shape=(N,)),
            motion=Unbounded(shape=(N,)),
            size=Unbounded(shape=(N,)),
            build_limit=Unbounded(shape=(1,)),
            current_node=Unbounded(shape=(1,)),
            i=Unbounded(shape=(1,)),
            assigned=Unbounded(shape=(N,)),
            open_group=Unbounded(shape=(N,)),
            group_count=Unbounded(shape=(1,)),
            action_mask=Unbounded(shape=(N,)),
            shape=(),
        )

        self.action_spec = Bounded(shape=(1,), dtype=torch.int64, low=0, high=N)
        self.reward_spec = Unbounded(shape=(1,))

    # =====================================================
    # ACTION MASK
    # =====================================================
    def get_action_mask(self, td: TensorDict) -> Tensor:

        compat = td["compat"]
        assigned = td["assigned"]
        open_group = td["open_group"]

        B, N, _ = compat.shape

        mask = ~assigned

        open_any = open_group.any(dim=-1, keepdim=True)

        if open_any.any():
            allowed = torch.ones((B, N), dtype=torch.bool, device=td.device)

            for idx in range(N):
                sel = open_group[:, idx]
                if sel.any():
                    allowed = torch.where(
                        sel.unsqueeze(-1),
                        allowed & compat[:, idx, :],
                        allowed,
                    )
            mask = mask & allowed

        all_assigned = assigned[..., 1:].all(dim=-1, keepdim=True)
        sep_allowed = open_any | all_assigned
        mask[:, 0] = sep_allowed.squeeze(-1)

        # safety: prevent all-False rows (NaN 방지)
        no_feasible = mask.sum(dim=-1) == 0
        if no_feasible.any():
            mask[no_feasible] = False
            mask[no_feasible, 0] = True

        return mask

    # =====================================================
    # REWARD
    # =====================================================
    def _get_reward(self, td: TensorDict, actions: Tensor) -> Tensor:

        B, T = actions.shape
        N = td["node_features"].shape[-2]

        material = td["material"]
        motion = td["motion"]
        size = td["size"]
        build_limit = td["build_limit"].squeeze(-1)

        group_count = torch.zeros((B,), device=actions.device)
        mat_viol = torch.zeros((B,), device=actions.device)
        mot_viol = torch.zeros((B,), device=actions.device)
        size_over = torch.zeros((B,), device=actions.device)

        open_members = [[] for _ in range(B)]
        open_size = torch.zeros((B,), device=actions.device)

        def close(b):
            if len(open_members[b]) == 0:
                return
            group_count[b] += 1

            mats = material[b, open_members[b]]
            if (mats != mats[0]).any():
                mat_viol[b] += 1

            mots = motion[b, open_members[b]]
            if (mots != mots[0]).any():
                mot_viol[b] += 1

            overflow = torch.relu(open_size[b] - build_limit[b])
            size_over[b] += overflow

            open_members[b] = []
            open_size[b] = 0

        assigned_parts = torch.zeros((B, N), dtype=torch.bool, device=actions.device)
        assigned_parts[:, 0] = True

        for t in range(T):
            a = actions[:, t]
            for b in range(B):
                if a[b] == 0:
                    close(b)
                else:
                    j = int(a[b])
                    if assigned_parts[b, j]:
                        continue
                    open_members[b].append(j)
                    open_size[b] += size[b, j]
                    assigned_parts[b, j] = True

        for b in range(B):
            close(b)

        cost = (
            group_count
            + self.w_material * mat_viol
            + self.w_motion * mot_viol
            + self.w_size * size_over
        )

        return -cost.unsqueeze(-1)
    
    def check_solution_validity(self, td: TensorDict, actions: Tensor) -> None:
        """
        Minimal validity check for RL4CO compatibility.
        We allow duplicates to be silently ignored by reward,
        so we do not raise errors here.
        """
        return
