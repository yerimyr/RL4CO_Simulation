"""
Train a prototype Part Consolidation policy with RL4CO's AttentionModel (REINFORCE).
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict

import torch
from torch.utils.tensorboard import SummaryWriter

from rl4co.envs.pc import PartConsolidationEnv
from rl4co.envs.pc.generator import PCGeneratorParams
from rl4co.models.zoo.am import AttentionModel
from rl4co.models.zoo.pc import make_pc_policy


def set_seed(seed: int = 1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(1234)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- TensorBoard ----------------
    writer = SummaryWriter(log_dir="runs/pc_experiment")

    # ---------- Problem / generator ----------
    gen_params = PCGeneratorParams(
        num_parts=4,
        extra_edge_prob=0.35,
        build_limit=1.6,
        size_low=0.1,
        size_high=1.0,
        material_types=3,
        motion_types=2,
    )

    env = PartConsolidationEnv(generator_params=asdict(gen_params)).to(device)

    # ---------- Model ----------
    policy = make_pc_policy(
        node_feat_dim=env.generator.node_feat_dim,
        embed_dim=128,
        num_encoder_layers=3,
    )

    model = AttentionModel(env, policy=policy, baseline="rollout").to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # ---------- Train loop ----------
    EPOCHS = 100000
    BATCH_SIZE = 20
    PRINT_EVERY = 100

    model.train()

    for epoch in range(1, EPOCHS + 1):

        td = env.reset(batch_size=BATCH_SIZE).to(device)
        out = model(td, decode_type="sampling")

        reward = out["reward"]
        log_likelihood = out["log_likelihood"]

        loss = -(reward.detach() * log_likelihood).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # -------- TensorBoard Logging --------
        writer.add_scalar("Train/Loss", loss.item(), epoch)
        writer.add_scalar("Train/Reward_sampled", reward.mean().item(), epoch)

        if epoch % PRINT_EVERY == 0:
            with torch.no_grad():
                td_val = env.reset(batch_size=256).to(device)
                out_greedy = model(td_val, decode_type="greedy")
                avg_reward = out_greedy["reward"].mean().item()

            print(
                f"Epoch {epoch:05d} | "
                f"loss={loss.item():.4f} | "
                f"greedy avg reward={avg_reward:.4f}"
            )

            writer.add_scalar("Validation/Reward_greedy", avg_reward, epoch)

    writer.close()

    # ---------- Demo ----------
    model.eval()
    td = env.reset(batch_size=1).to(device)
    out = model(td, decode_type="greedy")
    actions = out["actions"].squeeze(0).tolist()

    print("\n=== Demo (greedy) ===")
    print("material:", td["material"].squeeze(0).tolist())
    print("motion  :", td["motion"].squeeze(0).tolist())
    print("size    :", [round(x, 3) for x in td["size"].squeeze(0).tolist()])
    print("compat  :\n", td["compat"].squeeze(0).int().cpu().numpy())
    print("actions :", actions)
    print("reward  :", out["reward"].item())


if __name__ == "__main__":
    main()
