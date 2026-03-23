from __future__ import annotations
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.optim as optim

from rl4co.envs.pc.env import PartConsolidationEnv
from rl4co.envs.pc.generator import FPIGenerator
from rl4co.models.zoo.pc.policy import PCPolicy


def check_generator(td):
    print("\n===== CHECK 1: FPI NETWORK GENERATION =====")

    print("node_features shape:", td["node_features"].shape)
    print("edge_features shape:", td["edge_features"].shape)
    print("W shape:", td["W"].shape)
    print("compat shape:", td["compat"].shape)

    print("node_features sample:")
    print(td["node_features"][0])

    print("mat_var sample:")
    print(td["mat_var"][0])

    print("maint_diff sample:")
    print(td["maint_diff"][0])

    print("rel_motion sample:")
    print(td["rel_motion"][0])

    interaction_edges = (td["W"] > 0).sum()
    compat_edges = td["compat"].sum()

    print("interaction edges:", interaction_edges.item())
    print("compat edges:", compat_edges.item())

    fpi_only = ((td["W"] > 0) & (~td["compat"])).sum()
    print("interaction but infeasible edges:", fpi_only.item())


def check_mdp_state(td):
    print("\n===== CHECK 2: MDP STATE =====")

    print("assigned shape:", td["assigned"].shape)
    print("open_group shape:", td["open_group"].shape)
    print("action_mask shape:", td["action_mask"].shape)
    print("done shape:", td["done"].shape)

    print("assigned initial:", td["assigned"][0])
    print("open_group initial:", td["open_group"][0])
    print("action_mask initial:", td["action_mask"][0])


def check_policy_forward(policy, td):
    print("\n===== CHECK 3: ACTION SELECTION =====")

    with torch.no_grad():
        actions, logps = policy(td, max_steps=1, sample=False)

    print("actions:", actions)
    print("logps:", logps)

    assert actions.shape[0] == td.batch_size[0]
    assert logps.shape[0] == td.batch_size[0]

    print("action selection OK")


def check_policy_internal(policy, td):
    print("\n===== CHECK 4: POLICY NETWORK INTERNAL =====")

    with torch.no_grad():
        node_emb = policy.encode(td)

    print("node embedding shape:", node_emb.shape)

    logits = policy.compute_logits(node_emb, td)

    print("logits shape:", logits.shape)

    probs = torch.softmax(logits, dim=-1)

    print("probabilities sample:", probs[0])

    print("policy forward pipeline OK")


def check_env_step(env, td, policy):
    print("\n===== CHECK 5: ENV STEP / GROUP UPDATE =====")

    with torch.no_grad():
        actions, _ = policy(td, max_steps=1, sample=False)

    action = actions[:, 0]

    print("selected action:", action)

    td2 = env.step(td, action)

    print("assigned after step:", td2["assigned"][0])
    print("open_group after step:", td2["open_group"][0])
    print("new action_mask:", td2["action_mask"][0])

    return td2


def check_episode_termination(env, td, policy):
    print("\n===== CHECK 6: EPISODE TERMINATION =====")

    cur_td = td.clone()

    for i in range(50):

        with torch.no_grad():
            actions, _ = policy(cur_td, max_steps=1, sample=False)

        cur_td = env.step(cur_td, actions[:, 0])

        if cur_td["done"].all():
            print("episode terminated at step", i)
            break

    print("done flags:", cur_td["done"])

    assert cur_td["done"].all()

    return cur_td


def check_reward_and_pg(env, policy, td):
    print("\n===== CHECK 7: POLICY GRADIENT UPDATE =====")

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    cur_td = td.clone()

    actions_all = []
    logps_all = []

    for _ in range(20):

        actions, logps = policy(cur_td, max_steps=1, sample=True)

        actions_all.append(actions[:, 0])
        logps_all.append(logps[:, 0])

        cur_td = env.step(cur_td, actions[:, 0])

        if cur_td["done"].all():
            break

    actions_all = torch.stack(actions_all, dim=1)
    logps_all = torch.stack(logps_all, dim=1)

    reward = env.reward_from_actions(td["W"], actions_all)

    print("reward:", reward)

    baseline = reward.mean()
    advantage = reward - baseline

    logp_sum = logps_all.sum(dim=1)

    loss = -(advantage.detach() * logp_sum).mean()

    print("loss:", loss.item())

    optimizer.zero_grad()
    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)

    optimizer.step()

    print("gradient norm:", grad_norm)

    print("policy gradient update OK")


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    generator = FPIGenerator(num_parts=6)

    env = PartConsolidationEnv(
        generator=generator,
        reward_a=1.0,
        reward_b=0.2,
        device=device,
    )

    policy = PCPolicy(
        node_feat_dim=generator.node_feat_dim,
        edge_feat_dim=generator.edge_feat_dim,
        emb_dim=128,
        num_message_passing=3,
    ).to(device)

    td = env.reset(batch_size=1)

    td = td.to(device)

    check_generator(td)

    check_mdp_state(td)

    check_policy_forward(policy, td)

    check_policy_internal(policy, td)

    td2 = check_env_step(env, td, policy)

    td_end = check_episode_termination(env, td, policy)

    check_reward_and_pg(env, policy, td)

    print("\n===== VALIDATION COMPLETE =====")


if __name__ == "__main__":
    main()