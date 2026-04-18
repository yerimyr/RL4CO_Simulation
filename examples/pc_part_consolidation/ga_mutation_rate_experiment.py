from __future__ import annotations

import csv
import random
from pathlib import Path

from rl4co.envs.pc.env import PartConsolidationEnv
from rl4co.envs.pc.evaluator import evaluate_groups
from rl4co.envs.pc.generator import FPIGenerator
from rl4co.models.zoo.pc.ga_solver import GASolver

from examples.pc_part_consolidation.main import compute_search_space_proxy
from examples.pc_part_consolidation.main import save_instance_info_png
from examples.pc_part_consolidation.main import td_to_inst


def clone_env_with_num_parts(env: PartConsolidationEnv, num_parts: int) -> PartConsolidationEnv:
    generator_params = vars(env.generator.p).copy()
    generator_params["num_parts"] = int(num_parts)
    generator_params["max_num_parts"] = int(num_parts)
    generator = FPIGenerator(**generator_params)
    return PartConsolidationEnv(
        generator=generator,
        min_group_size_before_sep=env.min_group_size_before_sep,
        allow_fallback=env.allow_fallback,
        device=str(env.device),
    )


def generate_fixed_instance_set(
    env: PartConsolidationEnv,
    num_instances: int,
    min_parts: int,
    max_parts: int,
    seed: int,
):
    rng = random.Random(seed)
    instances = []
    for i in range(num_instances):
        sampled_num_parts = rng.randint(min_parts, max_parts)
        env_i = clone_env_with_num_parts(env, sampled_num_parts)
        td = env_i.reset(batch_size=1)
        inst = td_to_inst(td, env_i.generator.num_parts)
        instances.append(inst)
    return instances


def run_mutation_rate_experiment(
    env: PartConsolidationEnv,
    instances: list[dict],
    mutation_rates: list[float],
    output_dir: Path,
    ga_seed: int = 123,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for idx, inst in enumerate(instances):
        inst_dir = output_dir / "instances" / f"instance_{idx:03d}"
        save_instance_info_png(inst, f"MUTATION EXPERIMENT INSTANCE {idx}", inst_dir / "instance_info.png")

    for mutation_rate in mutation_rates:
        rate_tag = f"mr_{int(round(mutation_rate * 100)):02d}"
        rate_dir = output_dir / rate_tag
        rate_dir.mkdir(parents=True, exist_ok=True)

        for idx, inst in enumerate(instances):
            ga = GASolver(mutation_rate=mutation_rate, seed=ga_seed)
            grouping, elapsed = ga.solve(inst)
            ga.plot_fitness_history(str(rate_dir / f"instance_{idx:03d}_fitness.png"))

            metrics = evaluate_groups(grouping, inst)
            metrics["num_parts"] = inst["num_parts"]
            proxy = compute_search_space_proxy(inst)
            rows.append(
                {
                    "instance_id": idx,
                    "mutation_rate": mutation_rate,
                    "num_parts": inst["num_parts"],
                    "groups": len(grouping),
                    "grouping": str(grouping),
                    "time": elapsed,
                    "score": ga.last_best_score,
                    "feasible": metrics["feasible"],
                    "infeasible_solution": metrics["infeasible_solution"],
                    "infeasible_groups": metrics["infeasible_groups"],
                    "group_ratio": metrics["group_ratio"],
                    "total_internal_strength": metrics["total_internal_strength"],
                    "feasible_pair_count": metrics["feasible_pair_count"],
                    "normalized_internal_strength": metrics["normalized_internal_strength"],
                    "compat_density": proxy["compat_density"],
                    "feasible_pair_proxy": proxy["feasible_pair_proxy"],
                    "feasible_triple_proxy": proxy["feasible_triple_proxy"],
                    "feasible_quad_proxy": proxy["feasible_quad_proxy"],
                    "max_pairwise_clique_size": proxy["max_pairwise_clique_size"],
                }
            )
            print(
                f"[mutation-rate {mutation_rate:.2%}] "
                f"instance {idx + 1}/{len(instances)} done | "
                f"num_parts={inst['num_parts']} | groups={len(grouping)} | score={ga.last_best_score:.4f}"
            )

    return rows


def save_results(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ga_mutation_rate_results.csv"
    fieldnames = [
        "instance_id",
        "mutation_rate",
        "num_parts",
        "groups",
        "grouping",
        "time",
        "score",
        "feasible",
        "infeasible_solution",
        "infeasible_groups",
        "group_ratio",
        "total_internal_strength",
        "feasible_pair_count",
        "normalized_internal_strength",
        "compat_density",
        "feasible_pair_proxy",
        "feasible_triple_proxy",
        "feasible_quad_proxy",
        "max_pairwise_clique_size",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "ga_mutation_rate_summary.csv"
    summary_rows = []
    unique_rates = sorted({float(row["mutation_rate"]) for row in rows})
    for rate in unique_rates:
        items = [row for row in rows if float(row["mutation_rate"]) == rate]
        count = len(items)
        summary_rows.append(
            {
                "mutation_rate": rate,
                "instances": count,
                "avg_groups": sum(float(x["groups"]) for x in items) / count,
                "avg_time": sum(float(x["time"]) for x in items) / count,
                "avg_score": sum(float(x["score"]) for x in items) / count,
                "avg_group_ratio": sum(float(x["group_ratio"]) for x in items) / count,
                "avg_normalized_internal_strength": sum(float(x["normalized_internal_strength"]) for x in items) / count,
            }
        )

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mutation_rate",
                "instances",
                "avg_groups",
                "avg_time",
                "avg_score",
                "avg_group_ratio",
                "avg_normalized_internal_strength",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


def main():
    device = "cpu"
    generator_params = dict(
        num_parts=10,
        max_num_parts=10,
        material_types=1,
        p_relative_motion=0.01,
        p_extra_edge=0.80,
        L_low=5.0,
        L_high=160.0,
        W_low=5.0,
        W_high=70.0,
        H_low=0.5,
        H_high=30.0,
        build_limit_L=1600.0,
        build_limit_W=700.0,
        build_limit_H=300.0,
        p_maint_H=0.01,
        p_standard=0.01,
    )

    mutation_rates = [0.01, 0.05, 0.10, 0.20]
    num_instances = 10
    instance_seed = 123
    ga_seed = 123

    gen = FPIGenerator(**generator_params)
    env = PartConsolidationEnv(generator=gen, device=device)

    output_dir = Path("ga_mutation_rate_experiment")
    instances = generate_fixed_instance_set(
        env=env,
        num_instances=num_instances,
        min_parts=generator_params["num_parts"],
        max_parts=generator_params["max_num_parts"],
        seed=instance_seed,
    )
    rows = run_mutation_rate_experiment(
        env=env,
        instances=instances,
        mutation_rates=mutation_rates,
        output_dir=output_dir,
        ga_seed=ga_seed,
    )
    save_results(rows, output_dir)
    print(f"Saved results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
