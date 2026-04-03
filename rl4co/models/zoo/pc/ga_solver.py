from __future__ import annotations

import copy
import random
import time

import numpy as np
from deap import base
from deap import creator
from deap import tools

from rl4co.envs.pc.evaluator import DEFAULT_SCORE_WEIGHTS
from rl4co.envs.pc.evaluator import evaluate_groups
from rl4co.envs.pc.evaluator import score_metric_rows

class GASolver:
    """
    Constraint-aware GA baseline for part consolidation.

    Compared to the original toy implementation, this version:
    - scores solutions with the actual grouping constraints
    - keeps the global best solution with elitism
    - uses group-aware crossover
    - mutates with move / merge / split operators
    """

    def __init__(
        self,
        pop_size: int = 120,
        generations: int = 300,
        elite_size: int = 2,
        tournament_size: int = 3,
        mutation_rate: float = 0.85,
        init_new_group_bias: float = 0.35,
        enable_post_merge_repair: bool = False,
        seed: int | None = None,
    ):
        self.pop_size = int(pop_size)
        self.generations = int(generations)
        self.elite_size = int(elite_size)
        self.tournament_size = int(tournament_size)
        self.mutation_rate = float(mutation_rate)
        self.init_new_group_bias = float(init_new_group_bias)
        self.enable_post_merge_repair = bool(enable_post_merge_repair)
        self.rng = random.Random(seed)
        self.last_best_score: float | None = None
        self.last_generation_best_scores: list[float] = []
        self.last_generation_mean_scores: list[float] = []
        self.last_generation_best_raw_scores: list[float] = []
        self.last_generation_mean_raw_scores: list[float] = []
        self.score_weights = dict(DEFAULT_SCORE_WEIGHTS)

    def solve(self, inst):
        start = time.time()
        n = int(inst["num_parts"])
        toolbox = self._build_toolbox(inst, n)
        pop = toolbox.population(n=self.pop_size)
        self._evaluate_invalid(pop, toolbox)

        scores = self._population_scores([self._as_array(ind) for ind in pop], inst)
        raw_scores = [float(ind.fitness.values[0]) for ind in pop]

        best_idx = int(np.argmax(scores))
        best_sol = self._as_array(pop[best_idx]).copy()
        best_score = float(scores[best_idx])
        self.last_generation_best_scores = [best_score]
        self.last_generation_mean_scores = [float(np.mean(scores))]
        self.last_generation_best_raw_scores = [float(np.max(raw_scores))]
        self.last_generation_mean_raw_scores = [float(np.mean(raw_scores))]

        cxpb = 0.9
        for _ in range(self.generations):
            elites = [toolbox.clone(ind) for ind in tools.selBest(pop, self.elite_size)]
            offspring = tools.selTournament(pop, self.pop_size - self.elite_size, tournsize=self.tournament_size)
            offspring = [toolbox.clone(ind) for ind in offspring]

            for i in range(0, len(offspring) - 1, 2):
                if self.rng.random() < cxpb:
                    toolbox.mate(offspring[i], offspring[i + 1])
                    if offspring[i].fitness.valid:
                        del offspring[i].fitness.values
                    if offspring[i + 1].fitness.valid:
                        del offspring[i + 1].fitness.values

            for ind in offspring:
                if self.rng.random() < self.mutation_rate:
                    toolbox.mutate(ind)
                    if ind.fitness.valid:
                        del ind.fitness.values

            self._evaluate_invalid(offspring, toolbox)
            pop = elites + offspring

            scores = self._population_scores([self._as_array(ind) for ind in pop], inst)
            raw_scores = [float(ind.fitness.values[0]) for ind in pop]

            gen_best_idx = int(np.argmax(scores))
            gen_best_score = float(scores[gen_best_idx])
            self.last_generation_best_scores.append(gen_best_score)
            self.last_generation_mean_scores.append(float(np.mean(scores)))
            self.last_generation_best_raw_scores.append(float(np.max(raw_scores)))
            self.last_generation_mean_raw_scores.append(float(np.mean(raw_scores)))
            if gen_best_score > best_score:
                best_score = gen_best_score
                best_sol = self._as_array(pop[gen_best_idx]).copy()

        self.last_best_score = best_score
        end = time.time()
        return self._decode(best_sol), end - start

    def _build_toolbox(self, inst, n: int) -> base.Toolbox:
        if not hasattr(creator, "PCFitnessMax"):
            creator.create("PCFitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "PCIndividual"):
            creator.create("PCIndividual", list, fitness=creator.PCFitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("individual", self._make_individual, inst, n)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._evaluate_individual, inst=inst)
        toolbox.register("mate", self._mate_individuals, n=n, inst=inst)
        toolbox.register("mutate", self._mutate_individual, inst=inst)
        toolbox.register("clone", copy.deepcopy)
        return toolbox

    def _make_individual(self, inst, n: int):
        sol = self._random_solution(inst)
        return creator.PCIndividual(sol.tolist())

    def _evaluate_invalid(self, pop, toolbox) -> None:
        invalid = [ind for ind in pop if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

    def _evaluate_individual(self, ind, inst):
        return (self._fitness(self._as_array(ind), inst),)

    def _mate_individuals(self, ind1, ind2, n: int, inst):
        child1 = self._repair(self._crossover(self._as_array(ind1), self._as_array(ind2), n), inst)
        child2 = self._repair(self._crossover(self._as_array(ind2), self._as_array(ind1), n), inst)
        ind1[:] = child1.tolist()
        ind2[:] = child2.tolist()
        return ind1, ind2

    def _mutate_individual(self, ind, inst):
        child = self._repair(self._mutate(self._as_array(ind), inst), inst)
        ind[:] = child.tolist()
        return (ind,)

    def _as_array(self, sol) -> np.ndarray:
        if isinstance(sol, np.ndarray):
            return sol.astype(int, copy=True)
        return np.asarray(list(sol), dtype=int)

    def plot_fitness_history(self, save_path: str = "ga_fitness_history.png", show: bool = False) -> str:
        if not self.last_generation_best_scores:
            raise RuntimeError("No GA fitness history available. Run solve(...) first.")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        generations = list(range(len(self.last_generation_best_scores)))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        axes[0].plot(generations, self.last_generation_best_scores, label="Best Fitness", linewidth=2)
        axes[0].plot(generations, self.last_generation_mean_scores, label="Mean Fitness", linewidth=1.8)
        axes[0].set_xlabel("Generation")
        axes[0].set_ylabel("Fitness")
        axes[0].set_title("Normalized Fitness by Generation")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(generations, self.last_generation_best_raw_scores, label="Best Raw Fitness", linewidth=2)
        axes[1].plot(generations, self.last_generation_mean_raw_scores, label="Mean Raw Fitness", linewidth=1.8)
        axes[1].set_xlabel("Generation")
        axes[1].set_ylabel("Raw Fitness")
        axes[1].set_title("Raw Fitness by Generation")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        fig.suptitle("GA Fitness by Generation")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)
        return save_path

    def _random_solution(self, inst) -> np.ndarray:
        n = int(inst["num_parts"])
        order = list(range(n))
        self.rng.shuffle(order)

        groups: list[list[int]] = []
        for node in order:
            feasible_targets = []
            for idx in range(len(groups)):
                trial = sorted(groups[idx] + [node])
                if self._group_feasible(trial, inst):
                    feasible_targets.append(idx)

            create_new_group = (
                not feasible_targets
                or self.rng.random() < self.init_new_group_bias
            )
            if create_new_group:
                groups.append([node])
                continue

            target_idx = self.rng.choice(feasible_targets)
            groups[target_idx].append(node)

        return self._encode(groups, n)

    def _fitness(self, sol, inst) -> float:
        groups = self._decode(sol)
        metrics = evaluate_groups(groups, inst)
        # Fallback raw score if population-level normalization is not available.
        return (
            self.score_weights["infeasible_solution"] * metrics["infeasible_solution"]
            + self.score_weights["infeasible_groups"] * metrics["infeasible_groups"]
            + self.score_weights["num_groups"] * metrics["num_groups"]
            + self.score_weights["total_internal_strength"] * metrics["total_internal_strength"]
            + self.score_weights["feasible_pair_count"] * metrics["feasible_pair_count"]
        )

    def _population_scores(self, pop, inst) -> list[float]:
        rows = []
        for idx, sol in enumerate(pop):
            groups = self._decode(sol)
            metrics = evaluate_groups(groups, inst)
            metrics["idx"] = idx
            rows.append(metrics)

        scored = score_metric_rows(rows, weights=self.score_weights)
        scored.sort(key=lambda x: x["idx"])
        return [float(row["score"]) for row in scored]

    def _group_penalty(self, group: list[int], inst) -> float:
        penalty = 0.0
        compat = np.asarray(inst.get("compat", np.ones_like(inst["assembly_adj"])))

        if not self._group_size_ok(group, inst):
            penalty += 50.0

        for node in group:
            if not self._node_feasible(node, inst):
                penalty += 25.0

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                u, v = group[i], group[j]
                if compat[u, v] == 0:
                    penalty += 40.0

        if not self._connected(group, inst):
            penalty += 60.0

        return penalty

    def _decode(self, sol: np.ndarray) -> list[list[int]]:
        groups = {}
        for i, g in enumerate(sol):
            groups.setdefault(int(g), []).append(i)
        return [sorted(group) for group in groups.values()]

    def _encode(self, groups: list[list[int]], n: int) -> np.ndarray:
        sol = np.empty(n, dtype=int)
        for gid, group in enumerate(groups):
            for node in group:
                sol[node] = gid
        return sol

    def _canonicalize(self, sol: np.ndarray) -> np.ndarray:
        mapping = {}
        next_gid = 0
        out = np.empty_like(sol)
        for i, gid in enumerate(sol):
            gid = int(gid)
            if gid not in mapping:
                mapping[gid] = next_gid
                next_gid += 1
            out[i] = mapping[gid]
        return out

    def _crossover(self, p1: np.ndarray, p2: np.ndarray, n: int) -> np.ndarray:
        child = p1.copy()
        parent2_groups = self._decode(p2)
        self.rng.shuffle(parent2_groups)

        # Inherit a few entire groups from parent 2.
        take_k = max(1, len(parent2_groups) // 2)
        selected = parent2_groups[:take_k]
        for group in selected:
            new_gid = int(child.max()) + 1 if child.size > 0 else 0
            for node in group:
                child[node] = new_gid

        # Also align some nodes to parent 2 directly.
        for node in range(n):
            if self.rng.random() < 0.2:
                target_group = p2[node]
                members = np.where(p2 == target_group)[0]
                new_gid = int(child.max()) + 1
                child[members] = new_gid

        return self._canonicalize(child)

    def _mutate(self, sol: np.ndarray, inst) -> np.ndarray:
        child = sol.copy()
        if self.rng.random() > self.mutation_rate:
            return child

        op = self.rng.choice(["move", "merge", "split"])
        groups = self._decode(child)
        n = len(child)

        if op == "move":
            node = self.rng.randrange(n)
            current_gid = int(child[node])
            target_gids = [gid for gid in range(len(groups)) if gid != current_gid]
            self.rng.shuffle(target_gids)
            for gid in target_gids:
                trial = child.copy()
                trial[node] = gid
                if self._lightweight_valid_move(trial, inst):
                    return self._canonicalize(trial)

        if op == "merge" and len(groups) >= 2:
            pair_ids = [(i, j) for i in range(len(groups)) for j in range(i + 1, len(groups))]
            self.rng.shuffle(pair_ids)
            for i, j in pair_ids:
                merged = sorted(groups[i] + groups[j])
                if self._group_feasible(merged, inst):
                    trial = child.copy()
                    for node in groups[j]:
                        trial[node] = i
                    return self._canonicalize(trial)

        if op == "split":
            large_groups = [group for group in groups if len(group) >= 3]
            self.rng.shuffle(large_groups)
            for group in large_groups:
                cut = self.rng.randint(1, len(group) - 1)
                left = sorted(group[:cut])
                right = sorted(group[cut:])
                if self._group_feasible(left, inst) and self._group_feasible(right, inst):
                    trial = child.copy()
                    new_gid = int(child.max()) + 1
                    for node in right:
                        trial[node] = new_gid
                    return self._canonicalize(trial)

        return child

    def _repair(self, sol: np.ndarray, inst) -> np.ndarray:
        groups = self._decode(self._canonicalize(sol))
        repaired: list[list[int]] = []

        for group in sorted(groups, key=len, reverse=True):
            pending = list(group)
            while pending:
                placed_any = False
                for idx, node in enumerate(list(pending)):
                    candidate = [node]
                    for existing in list(pending):
                        if existing == node:
                            continue
                        trial = sorted(candidate + [existing])
                        if self._group_feasible(trial, inst):
                            candidate = trial
                    for node_in_candidate in candidate:
                        if node_in_candidate in pending:
                            pending.remove(node_in_candidate)
                    repaired.append(candidate)
                    placed_any = True
                    break
                if not placed_any:
                    repaired.extend([[node] for node in pending])
                    pending.clear()

        # Keep repair focused on feasibility recovery. Optional post-merge can be
        # enabled explicitly, but defaults off to preserve population diversity.
        if self.enable_post_merge_repair:
            improved = True
            while improved:
                improved = False
                best_pair = None
                best_gain = float("-inf")
                for i in range(len(repaired)):
                    for j in range(i + 1, len(repaired)):
                        merged = sorted(repaired[i] + repaired[j])
                        if not self._group_feasible(merged, inst):
                            continue
                        gain = self._internal_weight(merged, np.asarray(inst["W"], dtype=float))
                        if gain > best_gain:
                            best_gain = gain
                            best_pair = (i, j)
                if best_pair is not None:
                    i, j = best_pair
                    repaired[i] = sorted(repaired[i] + repaired[j])
                    repaired.pop(j)
                    improved = True

        return self._encode(repaired, len(sol))

    def _lightweight_valid_move(self, sol: np.ndarray, inst) -> bool:
        groups = self._decode(self._canonicalize(sol))
        return all(self._group_feasible(group, inst) for group in groups)

    def _internal_weight(self, group: list[int], w: np.ndarray) -> float:
        total = 0.0
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                total += float(w[group[i], group[j]])
        return total

    def _node_feasible(self, node: int, inst) -> bool:
        if "material_available" in inst and not np.asarray(inst["material_available"])[node]:
            return False

        size = np.asarray(inst["size"])
        build_limit = np.asarray(inst["build_limit"])
        if size.ndim == 1:
            return bool(size[node] <= build_limit)
        return bool(np.all(size[node] <= build_limit))

    def _group_size_ok(self, group: list[int], inst) -> bool:
        size = np.asarray(inst["size"])
        build_limit = np.asarray(inst["build_limit"])
        if size.ndim == 1:
            return bool(np.sum(size[group]) <= build_limit)
        return bool(np.all(np.sum(size[group], axis=0) <= build_limit))

    def _connected(self, group: list[int], inst) -> bool:
        if not group:
            return True
        adj = np.asarray(inst["assembly_adj"])
        visited = {group[0]}
        stack = [group[0]]
        while stack:
            cur = stack.pop()
            for nxt in group:
                if adj[cur, nxt] and nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        return len(visited) == len(group)

    def _group_feasible(self, group: list[int], inst) -> bool:
        compat = np.asarray(inst.get("compat", np.ones_like(inst["assembly_adj"])))
        if any(not self._node_feasible(node, inst) for node in group):
            return False
        if len(group) >= 2 and "isstandard" in inst and np.asarray(inst["isstandard"])[group].any():
            return False
        if not self._group_size_ok(group, inst):
            return False
        for i in group:
            for j in group:
                if compat[i, j] == 0:
                    return False
        return self._connected(group, inst)

    def _check_r3(self, groups: list[list[int]], inst):
        checker = inst.get("assembly_access_checker")
        if checker is None:
            return None
        for group in groups:
            ok, detail = checker(group, groups, inst)
            if not ok:
                return detail
        return None
