from __future__ import annotations

from collections import defaultdict

import numpy as np


DEFAULT_SCORE_WEIGHTS = {
    "infeasible_solution": -3.0,
    "infeasible_groups": -2.0,
    "num_groups": -1.5,
    "total_internal_strength": 1.0,
    "feasible_pair_count": 0.5,
}


def group_size_ok(group: list[int], inst) -> bool:
    size = np.asarray(inst["size"])
    build_limit = np.asarray(inst["build_limit"])
    if size.ndim == 1:
        return bool(np.sum(size[group]) <= build_limit)
    return bool(np.all(np.sum(size[group], axis=0) <= build_limit))


def node_feasible(node: int, inst) -> bool:
    if "isstandard" in inst and np.asarray(inst["isstandard"])[node]:
        return False
    if "material_available" in inst and not np.asarray(inst["material_available"])[node]:
        return False

    size = np.asarray(inst["size"])
    build_limit = np.asarray(inst["build_limit"])
    if size.ndim == 1:
        return bool(size[node] <= build_limit)
    return bool(np.all(size[node] <= build_limit))


def connected(group: list[int], inst) -> bool:
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


def group_feasible(group: list[int], inst) -> bool:
    compat = np.asarray(inst.get("compat", np.ones_like(inst["assembly_adj"])))
    if any(not node_feasible(node, inst) for node in group):
        return False
    if not group_size_ok(group, inst):
        return False
    for i in group:
        for j in group:
            if compat[i, j] == 0:
                return False
    return connected(group, inst)


def internal_strength(group: list[int], inst) -> float:
    w = np.asarray(inst["W"], dtype=float)
    total = 0.0
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            total += float(w[group[i], group[j]])
    return total


def feasible_pair_count(group: list[int], inst) -> int:
    compat = np.asarray(inst.get("compat", np.ones_like(inst["assembly_adj"])))
    count = 0
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            if compat[group[i], group[j]]:
                count += 1
    return count


def check_r3(groups: list[list[int]], inst) -> bool:
    checker = inst.get("assembly_access_checker")
    if checker is None:
        return True
    for group in groups:
        ok, _ = checker(group, groups, inst)
        if not ok:
            return False
    return True


def evaluate_groups(groups: list[list[int]], inst) -> dict[str, float]:
    infeasible_groups = 0
    total_internal_strength = 0.0
    total_feasible_pairs = 0

    for group in groups:
        feasible = group_feasible(group, inst)
        infeasible_groups += int(not feasible)
        total_internal_strength += internal_strength(group, inst)
        total_feasible_pairs += feasible_pair_count(group, inst)

    infeasible_solution = int(infeasible_groups > 0 or not check_r3(groups, inst))
    return {
        "feasible": float(1 - infeasible_solution),
        "infeasible_solution": float(infeasible_solution),
        "infeasible_groups": float(infeasible_groups),
        "num_groups": float(len(groups)),
        "total_internal_strength": float(total_internal_strength),
        "feasible_pair_count": float(total_feasible_pairs),
    }


def zscore_normalize_dicts(rows: list[dict], fields: list[str]) -> list[dict]:
    if not rows:
        return []

    stats = {}
    for field in fields:
        values = np.asarray([float(row[field]) for row in rows], dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if std < 1e-8:
            std = 1.0
        stats[field] = (mean, std)

    normalized = []
    for row in rows:
        new_row = dict(row)
        for field in fields:
            mean, std = stats[field]
            new_row[f"{field}_z"] = (float(row[field]) - mean) / std
        normalized.append(new_row)
    return normalized


def score_metric_rows(rows: list[dict], weights: dict[str, float] | None = None) -> list[dict]:
    weights = weights or DEFAULT_SCORE_WEIGHTS
    metric_fields = list(weights.keys())
    normalized = zscore_normalize_dicts(rows, metric_fields)

    scored = []
    for row in normalized:
        score = 0.0
        for field, weight in weights.items():
            score += weight * float(row[f"{field}_z"])
        out = dict(row)
        out["score"] = score
        scored.append(out)
    return scored


def score_metric_rows_by_group(
    rows: list[dict],
    group_fields: list[str],
    weights: dict[str, float] | None = None,
) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped[key].append(row)

    scored = []
    for _, items in grouped.items():
        scored.extend(score_metric_rows(items, weights=weights))
    return scored
