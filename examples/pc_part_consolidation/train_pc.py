from __future__ import annotations

import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np

from rl4co.envs.pc.env import PartConsolidationEnv
from rl4co.envs.pc.generator import FPIGenerator
from rl4co.models.zoo.pc.policy import PCPolicy


def rollout_episode_from_td(
    env: PartConsolidationEnv,
    policy: PCPolicy,
    td_init,
    max_steps: int,
    sample: bool = True,
):
    td = td_init.clone().to(env.device)

    actions = []
    logps = []
    entropies = []

    for _ in range(max_steps):
        action, logp, logits = policy.act(td, sample=sample)

        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)

        actions.append(action)
        logps.append(logp)
        entropies.append(entropy)

        td = env.step(td, action)
        if td["done"].all():
            break

    actions = torch.stack(actions, dim=1)
    logps = torch.stack(logps, dim=1)
    entropies = torch.stack(entropies, dim=1)

    reward = env.reward_from_actions(td["W"], actions)

    return actions, logps, entropies, reward, td


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # =========================
    # Hyperparameters
    # =========================
    batch_size = 128
    max_steps = 40
    epochs = 2000
    lr = 3e-4

    reward_a = 1.0
    reward_b = 0.2

    # =========================
    # TensorBoard
    # =========================
    log_dir = f"runs/pc_{int(time.time())}"
    writer = SummaryWriter(log_dir=log_dir)
    print("TensorBoard log dir:", log_dir)

    # =========================
    # Environment / Model
    # =========================
    generator_params = dict(
        num_parts=13,
        p_relative_motion=0.15,
        p_edge=0.35,
        L_low=5.0,
        L_high=220.0,
        W_low=5.0,
        W_high=80.0,
        H_low=0.5,
        H_high=30.0,
        build_limit_L=250.0,
        build_limit_W=120.0,
        build_limit_H=80.0,
        p_maint_H=0.10,
        p_standard=0.08,
    )

    gen = FPIGenerator(**generator_params)

    env = PartConsolidationEnv(
        generator=gen,
        reward_a=reward_a,
        reward_b=reward_b,
        device=device,
    )

    policy = PCPolicy(
        node_feat_dim=gen.node_feat_dim,
        edge_feat_dim=gen.edge_feat_dim,
        emb_dim=128,
        num_message_passing=3,
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # =========================
    # Training Loop
    # =========================
    for ep in range(1, epochs + 1):
        policy.train()

        td0 = env.reset(batch_size).to(device)

        actions, logps, entropies, reward, _ = rollout_episode_from_td(
            env=env,
            policy=policy,
            td_init=td0,
            max_steps=max_steps,
            sample=True,
        )

        # ===== baseline (greedy) =====
        policy.eval()
        with torch.no_grad():
            _, _, _, reward_greedy, _ = rollout_episode_from_td(
                env=env,
                policy=policy,
                td_init=td0,
                max_steps=max_steps,
                sample=False,
            )
        policy.train()

        # ===== loss =====
        advantage = reward - reward_greedy
        logp_sum = logps.sum(dim=1)
        loss = -(advantage.detach() * logp_sum).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        # =========================
        # TensorBoard Scalar Logging
        # =========================
        writer.add_scalar("train/reward", reward.mean().item(), ep)
        writer.add_scalar("train/reward_greedy", reward_greedy.mean().item(), ep)
        writer.add_scalar("train/loss", loss.item(), ep)
        writer.add_scalar("train/advantage", advantage.mean().item(), ep)
        writer.add_scalar("train/entropy", entropies.mean().item(), ep)

        # =========================
        # Action Distribution (IMAGE)
        # =========================
        flat_actions = actions.reshape(-1).cpu().numpy()

        counts = np.zeros(gen.num_nodes)
        for a in range(gen.num_nodes):
            counts[a] = np.sum(flat_actions == a)

        probs = counts / counts.sum()

        fig, ax = plt.subplots()
        ax.bar(range(gen.num_nodes), probs)
        ax.set_title(f"Action Distribution (ep={ep})")
        ax.set_xlabel("Action Index")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)

        writer.add_figure("action_dist/bar", fig, ep)
        plt.close(fig)

        # =========================
        # Heatmap (선택 행동 흐름)
        # =========================
        actions_np = actions.cpu().numpy()

        fig2, ax2 = plt.subplots()
        im = ax2.imshow(actions_np, aspect="auto", cmap="viridis")
        ax2.set_title(f"Action Heatmap (ep={ep})")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Batch")

        writer.add_figure("action_dist/heatmap", fig2, ep)
        plt.close(fig2)

        # =========================
        # Evaluation
        # =========================
        if ep % 10 == 0:
            policy.eval()
            with torch.no_grad():
                td_eval = env.reset(batch_size=256).to(device)
                _, _, _, reward_eval, _ = rollout_episode_from_td(
                    env=env,
                    policy=policy,
                    td_init=td_eval,
                    max_steps=max_steps,
                    sample=False,
                )

            writer.add_scalar("eval/reward", reward_eval.mean().item(), ep)

            print(
                f"[{ep:5d}] "
                f"train_reward={reward.mean().item():.4f} "
                f"greedy={reward_greedy.mean().item():.4f} "
                f"eval={reward_eval.mean().item():.4f} "
                f"loss={loss.item():.4f}"
            )

    writer.close()


if __name__ == "__main__":
    main()