import os
from pathlib import Path
import csv
from collections import defaultdict
import random
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import time

import torch
from tensordict import TensorDict

from rl4co.envs.pc.env import PartConsolidationEnv
from rl4co.envs.pc.evaluator import evaluate_groups
from rl4co.envs.pc.evaluator import score_metric_rows_by_group
from rl4co.envs.pc.fixed_instance_benchmark import create_fixed_instance
from rl4co.envs.pc.generator import FPIGenerator
from rl4co.models.zoo.pc.cpccd_solver import CPCCDSolver
from rl4co.models.zoo.pc.ga_solver import GASolver
from rl4co.models.zoo.pc.policy import PCPolicy


def _topology_name(topology_id):
    names = [
        "chain",
        "star",
        "tree",
        "two_module_bridge",
        "dense_clustered",
        "sparse_random",
    ]
    if topology_id is None:
        return None
    topology_id = int(topology_id)
    if 0 <= topology_id < len(names):
        return names[topology_id]
    return f"topology_{topology_id}"


def visualize_grouping_solution(inst, groups, method, output_path, metrics=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = int(inst["num_parts"])
    adj = inst["assembly_adj"]
    w = inst["W"]
    isstandard = inst["isstandard"]

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if bool(adj[i, j]):
                graph.add_edge(i, j, weight=float(w[i, j]))

    node_to_group = {}
    for gid, group in enumerate(groups):
        for node in group:
            node_to_group[int(node)] = gid

    palette = plt.cm.get_cmap("tab10", max(len(groups), 1) + 1)
    node_colors = []
    border_colors = []
    labels = {}
    for node in range(n):
        gid = node_to_group.get(node, -1)
        node_colors.append(palette(gid % palette.N) if gid >= 0 else (0.75, 0.75, 0.75, 1.0))
        border_colors.append("crimson" if bool(isstandard[node]) else "black")
        labels[node] = f"{node}" + ("S" if bool(isstandard[node]) else "")

    intra_edges = []
    inter_edges = []
    for u, v in graph.edges():
        same_group = node_to_group.get(u, -1) >= 0 and node_to_group.get(u, -1) == node_to_group.get(v, -2)
        if same_group:
            intra_edges.append((u, v))
        else:
            inter_edges.append((u, v))

    if graph.number_of_edges() > 0:
        pos = nx.spring_layout(graph, seed=7)
    else:
        pos = {
            node: (np.cos(2 * np.pi * node / max(n, 1)), np.sin(2 * np.pi * node / max(n, 1)))
            for node in range(n)
        }

    fig, ax = plt.subplots(figsize=(9, 7))
    nx.draw_networkx_edges(graph, pos, edgelist=inter_edges, edge_color="#bdbdbd", width=1.3, alpha=0.85, ax=ax)
    nx.draw_networkx_edges(graph, pos, edgelist=intra_edges, edge_color="#1b7f5f", width=2.7, alpha=0.95, ax=ax)
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        edgecolors=border_colors,
        linewidths=2.0,
        node_size=900,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10, font_weight="bold", ax=ax)
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels={(u, v): f"{graph[u][v]['weight']:.2f}" for u, v in graph.edges()},
        font_size=8,
        ax=ax,
    )

    title_lines = [f"{method} | num_parts={n}"]
    topo = _topology_name(inst.get("topology_id"))
    if topo is not None:
        title_lines[0] += f" | topology={topo}"
    title_lines.append(f"groups={len(groups)} | grouping={groups}")
    if metrics is not None:
        title_lines.append(
            " | ".join(
                [
                    f"feasible={int(metrics['feasible'])}",
                    f"infeasible_solution={int(metrics['infeasible_solution'])}",
                    f"infeasible_groups={int(metrics['infeasible_groups'])}",
                    f"strength={metrics['total_internal_strength']:.2f}",
                    f"pairs={metrics['feasible_pair_count']:.2f}",
                ]
            )
        )
    ax.set_title("\n".join(title_lines), fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def actions_to_groups(actions):
    groups = []
    current = []
    used = set()

    for a in actions:
        a = int(a)
        if a == 0:
            if current:
                groups.append(current)
                current = []
        else:
            part_idx = a - 1
            if part_idx not in used:
                current.append(part_idx)
                used.add(part_idx)

    if current:
        groups.append(current)

    return groups


def run_nco(env, policy, td, max_steps=None):
    start = time.time()
    td = td.clone().to(env.device)
    actions = []
    max_steps = max_steps or (env.generator.num_parts * 2 + 4)

    policy.eval()
    with torch.no_grad():
        for _ in range(max_steps):
            if td["done"].all():
                break
            action, _, _, _ = policy.act(td, sample=False, epsilon=0.0)
            actions.append(int(action[0].item()))
            td = env.step(td, action)

    groups = actions_to_groups(actions)
    return groups, time.time() - start


def evaluate(groups):
    return len(groups)


def td_to_inst(td, num_parts):
    actual_num_parts = int(td.get("num_parts", torch.tensor([num_parts]))[0].item())
    return {
        "num_parts": actual_num_parts,
        "topology_id": int(td.get("topology_id", torch.tensor([-1]))[0].item()),
        "material": td["material"][0, 1 : actual_num_parts + 1].cpu().numpy(),
        "size": td["size"][0, 1 : actual_num_parts + 1].cpu().numpy(),
        "maintfreq": td["maintfreq"][0, 1 : actual_num_parts + 1].cpu().numpy(),
        "isstandard": td["isstandard"][0, 1 : actual_num_parts + 1].cpu().numpy(),
        "material_available": (td["material"][0, 1 : actual_num_parts + 1] >= 0).cpu().numpy(),
        "build_limit": td["build_limit"][0].cpu().numpy(),
        "assembly_adj": td["assembly_adj"][0, 1 : actual_num_parts + 1, 1 : actual_num_parts + 1].cpu().numpy(),
        "mat_var": td["mat_var"][0, 1 : actual_num_parts + 1, 1 : actual_num_parts + 1].cpu().numpy(),
        "stack_size": td["stack_size"][0, 1 : actual_num_parts + 1, 1 : actual_num_parts + 1, :].cpu().numpy(),
        "maint_diff": td["maint_diff"][0, 1 : actual_num_parts + 1, 1 : actual_num_parts + 1].cpu().numpy(),
        "rel_motion": td["rel_motion"][0, 1 : actual_num_parts + 1, 1 : actual_num_parts + 1].cpu().numpy(),
        "compat": td["compat"][0, 1 : actual_num_parts + 1, 1 : actual_num_parts + 1].cpu().numpy(),
        "W": td["W"][0, 1 : actual_num_parts + 1, 1 : actual_num_parts + 1].cpu().numpy(),
        "relation_valid": td["relation_valid"][0, 1 : actual_num_parts + 1, 1 : actual_num_parts + 1].cpu().numpy(),
        "relation_consistent": bool(td["relation_consistent"][0].item()),
    }


def inst_to_td(inst, env):
    device = env.device
    generator = env.generator
    num_parts = int(inst["num_parts"])
    num_nodes = num_parts + 1
    material_types = generator.p.material_types

    material = torch.as_tensor(inst["material"], dtype=torch.long, device=device)
    size = torch.as_tensor(inst["size"], dtype=torch.float32, device=device)
    maintfreq = torch.as_tensor(inst["maintfreq"], dtype=torch.long, device=device)
    isstandard = torch.as_tensor(inst["isstandard"], dtype=torch.long, device=device)
    build_limit = torch.as_tensor(inst["build_limit"], dtype=torch.float32, device=device)
    assembly_adj = torch.as_tensor(inst["assembly_adj"], dtype=torch.bool, device=device)
    mat_var = torch.as_tensor(inst["mat_var"], dtype=torch.float32, device=device)
    stack_size = torch.as_tensor(inst["stack_size"], dtype=torch.float32, device=device)
    maint_diff = torch.as_tensor(inst["maint_diff"], dtype=torch.float32, device=device)
    rel_motion = torch.as_tensor(inst["rel_motion"], dtype=torch.float32, device=device)
    compat = torch.as_tensor(inst["compat"], dtype=torch.bool, device=device)
    W = torch.as_tensor(inst["W"], dtype=torch.float32, device=device)
    relation_valid = torch.as_tensor(inst.get("relation_valid", inst["assembly_adj"]), dtype=torch.bool, device=device)

    degree = assembly_adj.sum(dim=-1).float()
    max_degree = degree.max().clamp_min(1.0)
    pos1d = (degree / max_degree).unsqueeze(-1)
    material_one_hot = torch.nn.functional.one_hot(material, num_classes=material_types).float()

    part_node_features = torch.cat(
        [
            material_one_hot,
            size,
            maintfreq.float().unsqueeze(-1),
            isstandard.float().unsqueeze(-1),
            pos1d,
        ],
        dim=-1,
    )
    part_edge_features = torch.cat(
        [
            assembly_adj.float().unsqueeze(-1),
            mat_var.unsqueeze(-1),
            stack_size,
            maint_diff.unsqueeze(-1),
            rel_motion.unsqueeze(-1),
        ],
        dim=-1,
    )

    node_features = torch.zeros((1, num_nodes, generator.node_feat_dim), dtype=torch.float32, device=device)
    node_features[:, 1:, :] = part_node_features.unsqueeze(0)

    edge_features = torch.zeros((1, num_nodes, num_nodes, generator.edge_feat_dim), dtype=torch.float32, device=device)
    edge_features[:, 1:, 1:, :] = part_edge_features.unsqueeze(0)

    material_all = torch.full((1, num_nodes), -1, dtype=torch.long, device=device)
    maint_all = torch.full((1, num_nodes), -1, dtype=torch.long, device=device)
    std_all = torch.full((1, num_nodes), -1, dtype=torch.long, device=device)
    size_all = torch.zeros((1, num_nodes, 3), dtype=torch.float32, device=device)
    pos_all = torch.zeros((1, num_nodes, 1), dtype=torch.float32, device=device)
    W_all = torch.zeros((1, num_nodes, num_nodes), dtype=torch.float32, device=device)
    assembly_all = torch.zeros((1, num_nodes, num_nodes), dtype=torch.bool, device=device)
    mat_var_all = torch.zeros((1, num_nodes, num_nodes), dtype=torch.float32, device=device)
    maint_diff_all = torch.zeros((1, num_nodes, num_nodes), dtype=torch.float32, device=device)
    rel_motion_all = torch.zeros((1, num_nodes, num_nodes), dtype=torch.float32, device=device)
    stack_all = torch.zeros((1, num_nodes, num_nodes, 3), dtype=torch.float32, device=device)
    compat_all = torch.ones((1, num_nodes, num_nodes), dtype=torch.bool, device=device)
    relation_valid_all = torch.zeros((1, num_nodes, num_nodes), dtype=torch.bool, device=device)

    material_all[:, 1:] = material.unsqueeze(0)
    maint_all[:, 1:] = maintfreq.unsqueeze(0)
    std_all[:, 1:] = isstandard.unsqueeze(0)
    size_all[:, 1:, :] = size.unsqueeze(0)
    pos_all[:, 1:, :] = pos1d.unsqueeze(0)
    W_all[:, 1:, 1:] = W.unsqueeze(0)
    assembly_all[:, 1:, 1:] = assembly_adj.unsqueeze(0)
    mat_var_all[:, 1:, 1:] = mat_var.unsqueeze(0)
    maint_diff_all[:, 1:, 1:] = maint_diff.unsqueeze(0)
    rel_motion_all[:, 1:, 1:] = rel_motion.unsqueeze(0)
    stack_all[:, 1:, 1:, :] = stack_size.unsqueeze(0)
    compat_all[:, 1:, 1:] = compat.unsqueeze(0)
    relation_valid_all[:, 1:, 1:] = relation_valid.unsqueeze(0)

    assigned = torch.zeros((1, num_nodes), dtype=torch.bool, device=device)
    assigned[:, 0] = True
    open_group = torch.zeros((1, num_nodes), dtype=torch.bool, device=device)
    open_group_size = torch.zeros((1, 3), dtype=torch.float32, device=device)
    closed_group_count = torch.zeros((1,), dtype=torch.long, device=device)

    td = TensorDict(
        {
            "node_features": node_features,
            "edge_features": edge_features,
            "material": material_all,
            "size": size_all,
            "maintfreq": maint_all,
            "isstandard": std_all,
            "pos1d": pos_all,
            "W": W_all,
            "assembly_adj": assembly_all,
            "mat_var": mat_var_all,
            "stack_size": stack_all,
            "maint_diff": maint_diff_all,
            "rel_motion": rel_motion_all,
            "compat": compat_all,
            "relation_valid": relation_valid_all,
            "relation_consistent": torch.tensor([bool(inst.get("relation_consistent", True))], dtype=torch.bool, device=device),
            "build_limit": build_limit.unsqueeze(0),
            "assigned": assigned,
            "open_group": open_group,
            "open_group_size": open_group_size,
            "closed_group_count": closed_group_count,
            "fallback_part_mask": torch.zeros((1, num_nodes), dtype=torch.bool, device=device),
            "dead_end": torch.zeros((1, 1), dtype=torch.bool, device=device),
            "done": torch.zeros((1, 1), dtype=torch.bool, device=device),
            "action_mask": torch.ones((1, num_nodes), dtype=torch.bool, device=device),
        },
        batch_size=[1],
    )
    td["action_mask"] = env.get_action_mask(td)
    td["dead_end"] = env._compute_dead_end(td["assigned"], td["open_group"], td["action_mask"])
    td["done"] = td["done"] | td["dead_end"]
    return td


def print_instance(inst, title):
    print(f"\n===== {title} =====")
    print("num_parts:")
    print(inst["num_parts"])
    print("build_limit [L, W, H]:")
    print(inst["build_limit"])
    print("material:")
    print(inst["material"])
    print("maintfreq:")
    print(inst["maintfreq"])
    print("isstandard:")
    print(inst["isstandard"])
    print("size [L, W, H] per part:")
    print(inst["size"])
    print("assembly_adj:")
    print(inst["assembly_adj"].astype(int))
    print("mat_var:")
    print(inst["mat_var"].astype(int))
    print("maint_diff:")
    print(inst["maint_diff"].astype(int))
    print("rel_motion:")
    print(inst["rel_motion"].astype(int))
    print("compat:")
    print(inst["compat"].astype(int))
    print("W:")
    print(np_round(inst["W"]))


def np_round(arr, decimals=3):
    import numpy as np

    return np.round(arr, decimals=decimals)


def result_row(instance_type, instance_id, method, groups, elapsed, metrics):
    group_count = evaluate(groups)
    return {
        "instance_type": instance_type,
        "instance_id": instance_id,
        "num_parts": int(metrics.get("num_parts", 0)),
        "method": method,
        "groups": group_count,
        "num_groups": group_count,
        "time": float(elapsed),
        "grouping": str(groups),
        **metrics,
    }


def summarize_result_rows(rows, label):
    prepared = []
    for row in rows:
        new_row = dict(row)
        if "num_groups" not in new_row and "groups" in new_row:
            new_row["num_groups"] = new_row["groups"]
        prepared.append(new_row)
    scored_rows = score_metric_rows_by_group(prepared, group_fields=["instance_type"])
    summary = summarize_results(scored_rows)
    print(f"\n===== {label} SUMMARY =====")
    for row in summary:
        print(row)
    return scored_rows, summary


def run_fixed(env, policy):
    inst = create_fixed_instance(num_parts=env.generator.num_parts)
    inst["num_parts"] = int(env.generator.num_parts)
    print_instance(inst, "FIXED INSTANCE USED IN THE EXPERIMENT")
    td = inst_to_td(inst, env)

    cpccd = CPCCDSolver()
    ga = GASolver()

    g1, t1 = cpccd.solve(inst)
    g2, t2 = ga.solve(inst)
    ga.plot_fitness_history("ga_fitness_fixed.png")
    g3, t3 = run_nco(env, policy, td)

    m1 = evaluate_groups(g1, inst)
    m2 = evaluate_groups(g2, inst)
    m3 = evaluate_groups(g3, inst)
    m1["num_parts"] = inst["num_parts"]
    m2["num_parts"] = inst["num_parts"]
    m3["num_parts"] = inst["num_parts"]

    visualize_grouping_solution(inst, g1, "CPCCD", Path("visualizations") / "fixed" / "cpccd.png", m1)
    visualize_grouping_solution(inst, g2, "GA", Path("visualizations") / "fixed" / "ga.png", m2)
    visualize_grouping_solution(inst, g3, "NCO", Path("visualizations") / "fixed" / "nco.png", m3)

    return inst, [
        result_row("fixed", 0, "CPCCD", g1, t1, m1),
        result_row("fixed", 0, "GA", g2, t2, m2),
        result_row("fixed", 0, "NCO", g3, t3, m3),
    ]


def _clone_env_with_num_parts(env, num_parts: int):
    generator_params = vars(env.generator.p).copy()
    generator_params["num_parts"] = int(num_parts)
    generator = FPIGenerator(**generator_params)
    return PartConsolidationEnv(
        generator=generator,
        min_group_size_before_sep=env.min_group_size_before_sep,
        allow_fallback=env.allow_fallback,
        device=str(env.device),
    )


def run_generalization(env, policy, num_instances=30, min_parts=4, max_parts=10, seed=123):
    results = []
    cpccd = CPCCDSolver()
    ga = GASolver()
    plot_dir = Path("ga_fitness_generalization")
    plot_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    for i in range(num_instances):
        sampled_num_parts = rng.randint(min_parts, max_parts)
        env_i = _clone_env_with_num_parts(env, sampled_num_parts)
        td = env_i.reset(batch_size=1)
        inst = td_to_inst(td, env_i.generator.num_parts)
        inst["num_parts"] = int(env_i.generator.num_parts)

        g1, t1 = cpccd.solve(inst)
        g2, t2 = ga.solve(inst)
        ga.plot_fitness_history(str(plot_dir / f"ga_fitness_instance_{i}.png"))
        g3, t3 = run_nco(env_i, policy, td)

        m1 = evaluate_groups(g1, inst)
        m1["num_parts"] = inst["num_parts"]
        m2 = evaluate_groups(g2, inst)
        m2["num_parts"] = inst["num_parts"]
        m3 = evaluate_groups(g3, inst)
        m3["num_parts"] = inst["num_parts"]

        vis_dir = Path("visualizations") / "generalization" / f"instance_{i:03d}"
        visualize_grouping_solution(inst, g1, "CPCCD", vis_dir / "cpccd.png", m1)
        visualize_grouping_solution(inst, g2, "GA", vis_dir / "ga.png", m2)
        visualize_grouping_solution(inst, g3, "NCO", vis_dir / "nco.png", m3)

        results.append(result_row("generalization", i, "CPCCD", g1, t1, m1))
        results.append(result_row("generalization", i, "GA", g2, t2, m2))
        results.append(result_row("generalization", i, "NCO", g3, t3, m3))

        print(f"[generalization {i + 1}/{num_instances}] done | num_parts={sampled_num_parts}")

    return results


def save_results(results, filename):
    prepared = []
    for row in results:
        new_row = dict(row)
        if "num_groups" not in new_row:
            new_row["num_groups"] = new_row["groups"]
        prepared.append(new_row)
    scored_results = score_metric_rows_by_group(prepared, group_fields=["instance_type"])
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "instance_type",
            "instance_id",
            "num_parts",
            "method",
            "grouping",
            "groups",
            "time",
            "feasible",
            "infeasible_solution",
            "infeasible_groups",
            "total_internal_strength",
            "feasible_pair_count",
            "score",
        ])
        for row in scored_results:
            writer.writerow([
                row["instance_type"],
                row["instance_id"],
                row.get("num_parts", ""),
                row["method"],
                row["grouping"],
                row["groups"],
                row["time"],
                row["feasible"],
                row["infeasible_solution"],
                row["infeasible_groups"],
                row["total_internal_strength"],
                row["feasible_pair_count"],
                row["score"],
            ])

    summary = summarize_results(scored_results)
    with open(filename.replace(".csv", "_summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "instance_type",
                "method",
                "groups",
                "time",
                "feasible",
                "infeasible_solution",
                "infeasible_groups",
                "total_internal_strength",
                "feasible_pair_count",
                "score",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)

    return scored_results, summary


def summarize_results(results):
    agg = defaultdict(
        lambda: {
            "groups": 0.0,
            "time": 0.0,
            "feasible": 0.0,
            "infeasible_solution": 0.0,
            "infeasible_groups": 0.0,
            "total_internal_strength": 0.0,
            "feasible_pair_count": 0.0,
            "score": 0.0,
            "count": 0,
        }
    )
    for row in results:
        instance_type = row["instance_type"]
        method = row["method"]
        key = (instance_type, method)
        groups = row["groups"]
        elapsed = row["time"]
        agg[key]["groups"] += float(groups)
        agg[key]["time"] += float(elapsed)
        agg[key]["feasible"] += float(row["feasible"])
        agg[key]["infeasible_solution"] += float(row["infeasible_solution"])
        agg[key]["infeasible_groups"] += float(row["infeasible_groups"])
        agg[key]["total_internal_strength"] += float(row["total_internal_strength"])
        agg[key]["feasible_pair_count"] += float(row["feasible_pair_count"])
        agg[key]["score"] += float(row["score"])
        agg[key]["count"] += 1

    summary = []
    for (instance_type, method), vals in sorted(agg.items()):
        count = max(vals["count"], 1)
        summary.append(
            {
                "instance_type": instance_type,
                "method": method,
                "groups": vals["groups"] / count,
                "time": vals["time"] / count,
                "feasible": vals["feasible"] / count,
                "infeasible_solution": vals["infeasible_solution"] / count,
                "infeasible_groups": vals["infeasible_groups"] / count,
                "total_internal_strength": vals["total_internal_strength"] / count,
                "feasible_pair_count": vals["feasible_pair_count"] / count,
                "score": vals["score"] / count,
            }
        )
    return summary


def plot_results(results, title, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        print(f"skip plot '{title}': {exc}")
        return

    grouped_groups = defaultdict(list)
    grouped_time = defaultdict(list)
    for row in results:
        grouped_groups[row["method"]].append(row["groups"])
        grouped_time[row["method"]].append(row["time"])

    methods = sorted(grouped_groups.keys())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].boxplot([grouped_groups[m] for m in methods], tick_labels=methods)
    axes[0].set_title("Group Count")
    axes[0].set_xlabel("Method")
    axes[0].set_ylabel("Groups")

    axes[1].boxplot([grouped_time[m] for m in methods], tick_labels=methods)
    axes[1].set_title("Runtime")
    axes[1].set_xlabel("Method")
    axes[1].set_ylabel("Seconds")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def load_policy(gen, device):
    policy = PCPolicy(
        node_feat_dim=gen.node_feat_dim,
        edge_feat_dim=gen.edge_feat_dim,
    ).to(device)

    ckpt_path = Path("C:\\RL4CO_Simulation\\checkpoints\\best_model.pt")  #################################
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    return policy


def main():
    device = "cpu"

    gen = FPIGenerator(num_parts=4)
    env = PartConsolidationEnv(generator=gen, device=device)
    policy = load_policy(gen, device)

    print("\n===== FIXED INSTANCE EXPERIMENT =====")
    fixed_inst, fixed_results = run_fixed(env, policy)
    df_fixed, summary_fixed = save_results(fixed_results, "fixed_results.csv")
    for row in summary_fixed:
        print(row)

    print("\n===== GENERALIZATION EXPERIMENT =====")
    gen_results = run_generalization(env, policy, num_instances=100)
    df_gen, summary_gen = save_results(gen_results, "generalization_results.csv")
    for row in summary_gen:
        print(row)

    all_results = fixed_results + gen_results
    df_all, summary_all = save_results(all_results, "all_results.csv")
    summarize_result_rows(all_results, "ALL")

    plot_results(fixed_results, "Fixed_Instance_Result", "Fixed_Instance_Result.png")
    plot_results(gen_results, "Generalization_Result", "Generalization_Result.png")
    plot_results(all_results, "All_Results", "All_Results.png")

    return {
        "fixed_instance": fixed_inst,
        "fixed_results": df_fixed,
        "fixed_summary": summary_fixed,
        "generalization_results": df_gen,
        "generalization_summary": summary_gen,
        "all_summary": summary_all,
    }


if __name__ == "__main__":
    main()
