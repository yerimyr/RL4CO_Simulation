from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# optional: allow running this file directly after replacing project files
PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from rl4co.envs.pc.env import PartConsolidationEnv
from rl4co.envs.pc.generator import FPIGenerator
from rl4co.models.zoo.pc.policy import PCPolicy


def rollout_episode_from_td(
    env: PartConsolidationEnv,
    policy: PCPolicy,
    td_init,
    max_steps: int,
    sample: bool = True,
    epsilon: float = 0.0,
):
    td = td_init.clone().to(env.device)

    actions = []
    logps = []
    entropies = []
    step_rewards = []

    for _ in range(max_steps):
        action, logp, entropy, _ = policy.act(td, sample=sample, epsilon=epsilon)
        actions.append(action)
        logps.append(logp)
        entropies.append(entropy)

        td = env.step(td, action)
        step_rewards.append(td["step_reward"])

        if td["done"].all():
            break

    actions = torch.stack(actions, dim=1)
    logps = torch.stack(logps, dim=1)
    entropies = torch.stack(entropies, dim=1)
    step_rewards = torch.stack(step_rewards, dim=1)

    terminal_reward = env.reward_from_actions(actions)
    total_reward = step_rewards.sum(dim=1) + terminal_reward

    return actions, logps, entropies, step_rewards, terminal_reward, total_reward, td


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # =========================
    # Hyperparameters
    # =========================
    batch_size = 64
    epochs = 3000
    lr = 1e-4
    entropy_coef = 0.03
    grad_clip = 1.0

    eps_start = 0.3
    eps_end = 0.05
    eps_decay_epochs = 5000

    # =========================
    # TensorBoard
    # =========================
    log_dir = f"runs/pc_general_graph_groupcount_{int(time.time())}"
    writer = SummaryWriter(log_dir=log_dir)
    print("TensorBoard log dir:", log_dir)

    # =========================
    # Environment / Model
    # =========================
    generator_params = dict(
        num_parts=4,
        material_types=4,
        p_relative_motion=0.15,
        p_extra_edge=0.30,
        L_low=5.0,
        L_high=220.0,
        W_low=5.0,
        W_high=80.0,
        H_low=0.5,
        H_high=30.0,
        build_limit_L=260.0,
        build_limit_W=120.0,
        build_limit_H=80.0,
        p_maint_H=0.20,
        p_standard=0.10,
    )

    gen = FPIGenerator(**generator_params)
    env = PartConsolidationEnv(
        generator=gen,
        min_group_size_before_sep=1,
        device=device,
    )

    policy = PCPolicy(
        node_feat_dim=gen.node_feat_dim,
        edge_feat_dim=gen.edge_feat_dim,
        emb_dim=128,
        num_message_passing=3,
        temperature=1.2,
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    max_steps = gen.num_parts * 2 + 4

    # =========================
    # Training Loop
    # =========================
    for ep in range(1, epochs + 1):
        policy.train()

        frac = min(1.0, ep / float(eps_decay_epochs))
        epsilon = eps_start + (eps_end - eps_start) * frac

        td0 = env.reset(batch_size).to(device)
        actions, logps, entropies, step_rewards, terminal_reward, total_reward, _ = rollout_episode_from_td(
            env=env,
            policy=policy,
            td_init=td0,
            max_steps=max_steps,
            sample=True,
            epsilon=epsilon,
        )

        # greedy baseline on the SAME initial instances
        policy.eval()
        with torch.no_grad():
            _, _, _, _, _, reward_greedy, _ = rollout_episode_from_td(
                env=env,
                policy=policy,
                td_init=td0,
                max_steps=max_steps,
                sample=False,
                epsilon=0.0,
            )
        policy.train()

        advantage = total_reward - reward_greedy
        logp_sum = logps.sum(dim=1)
        entropy_mean = entropies.mean()

        loss_pg = -(advantage.detach() * logp_sum).mean()
        loss = loss_pg - entropy_coef * entropy_mean

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        optimizer.step()

        groups = env.actions_to_groups(actions, N=gen.num_nodes)
        avg_group_count = float(np.mean([len(g) for g in groups]))
        avg_group_size = float(
            np.mean([np.mean([len(x) for x in g]) if len(g) > 0 else 0.0 for g in groups])
        )
        avg_step_reward = step_rewards.mean().item()
        avg_terminal_reward = terminal_reward.mean().item()

        writer.add_scalar("train/reward_total", total_reward.mean().item(), ep)
        writer.add_scalar("train/reward_greedy", reward_greedy.mean().item(), ep)
        writer.add_scalar("train/reward_step_mean", avg_step_reward, ep)
        writer.add_scalar("train/reward_terminal_mean", avg_terminal_reward, ep)
        writer.add_scalar("train/loss", loss.item(), ep)
        writer.add_scalar("train/loss_pg", loss_pg.item(), ep)
        writer.add_scalar("train/advantage", advantage.mean().item(), ep)
        writer.add_scalar("train/entropy", entropy_mean.item(), ep)
        writer.add_scalar("train/epsilon", epsilon, ep)
        writer.add_scalar("train/avg_group_count", avg_group_count, ep)
        writer.add_scalar("train/avg_group_size", avg_group_size, ep)

        flat_actions = actions.reshape(-1).detach().cpu().numpy()
        counts = np.zeros(gen.num_nodes)
        for a in range(gen.num_nodes):
            counts[a] = np.sum(flat_actions == a)
        probs = counts / max(counts.sum(), 1.0)

        fig, ax = plt.subplots()
        ax.bar(range(gen.num_nodes), probs)
        ax.set_title(f"Action Distribution (ep={ep})")
        ax.set_xlabel("Action Index")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        writer.add_figure("action_dist/bar", fig, ep)
        plt.close(fig)

        actions_np = actions.detach().cpu().numpy()
        fig2, ax2 = plt.subplots()
        ax2.imshow(actions_np, aspect="auto", cmap="viridis")
        ax2.set_title(f"Action Heatmap (ep={ep})")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Batch")
        writer.add_figure("action_dist/heatmap", fig2, ep)
        plt.close(fig2)

        if ep % 10 == 0:
            policy.eval()
            with torch.no_grad():
                td_eval = env.reset(batch_size=256).to(device)
                _, _, _, _, _, reward_eval, _ = rollout_episode_from_td(
                    env=env,
                    policy=policy,
                    td_init=td_eval,
                    max_steps=max_steps,
                    sample=False,
                    epsilon=0.0,
                )
            writer.add_scalar("eval/reward_total", reward_eval.mean().item(), ep)

            print(
                f"[{ep:5d}] "
                f"train_total={total_reward.mean().item():.4f} "
                f"greedy={reward_greedy.mean().item():.4f} "
                f"eval_total={reward_eval.mean().item():.4f} "
                f"loss={loss.item():.4f} "
                f"entropy={entropy_mean.item():.4f} "
                f"avg_group_count={avg_group_count:.2f} "
                f"avg_group_size={avg_group_size:.2f} "
                f"eps={epsilon:.3f}"
            )

    writer.close()


if __name__ == "__main__":
    main()
