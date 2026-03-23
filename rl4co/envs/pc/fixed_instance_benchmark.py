import numpy as np


def create_fixed_instance(
    num_parts=4,
    build_limit=(260.0, 120.0, 80.0),
    seed=42,
):
    rng = np.random.default_rng(seed)

    # Match generator.py semantics:
    # - per-node size is a 3D vector [L, W, H]
    # - build_limit is also a 3D vector
    # - pairwise stack_size keeps the same [L, W, H] shape
    build_limit = np.asarray(build_limit, dtype=float)
    if build_limit.shape != (3,):
        raise ValueError("build_limit must be a length-3 iterable: (L_limit, W_limit, H_limit)")

    material = np.array([(i // 2) % 2 for i in range(num_parts)], dtype=int)
    maintfreq = np.array([(i % 3) == 0 for i in range(num_parts)], dtype=int)
    isstandard = np.zeros(num_parts, dtype=int)
    material_available = np.ones(num_parts, dtype=bool)

    L = rng.uniform(5.0, 160.0, size=num_parts)
    W = rng.uniform(5.0, 70.0, size=num_parts)
    H = rng.uniform(5.0, 30.0, size=num_parts)
    size = np.stack([L, W, H], axis=-1).astype(float)

    assembly_adj = np.zeros((num_parts, num_parts), dtype=bool)
    for i in range(num_parts - 1):
        assembly_adj[i, i + 1] = True
        assembly_adj[i + 1, i] = True

    eye = np.eye(num_parts, dtype=bool)

    mat_var = (material[:, None] != material[None, :]) & assembly_adj
    maint_diff = (maintfreq[:, None] != maintfreq[None, :]) & assembly_adj

    rel_motion = np.zeros((num_parts, num_parts), dtype=bool)
    if num_parts >= 4:
        rel_motion[2, 3] = True
        rel_motion[3, 2] = True
    rel_motion &= assembly_adj

    stack_size_full = size[:, None, :] + size[None, :, :]
    stack_size = stack_size_full * assembly_adj[:, :, None].astype(float)
    stack_ok = (stack_size_full <= build_limit.reshape(1, 1, 3)).all(axis=-1)

    standard_pair_block = isstandard[:, None].astype(bool) | isstandard[None, :].astype(bool)
    compat = (
        assembly_adj
        & (~mat_var)
        & (~maint_diff)
        & (~rel_motion)
        & stack_ok
        & (~standard_pair_block)
    )
    compat |= eye

    alpha = 0.4
    beta = 0.3
    gamma = 0.8
    score = 1.0 - alpha * mat_var.astype(float) - beta * maint_diff.astype(float) - gamma * rel_motion.astype(float)
    score = np.clip(score, 0.1, 1.0)
    W_matrix = assembly_adj.astype(float) * score
    np.fill_diagonal(W_matrix, 0.0)

    return {
        "num_parts": num_parts,
        "material": material,
        "size": size,
        "maintfreq": maintfreq,
        "isstandard": isstandard,
        "material_available": material_available,
        "build_limit": build_limit,
        "assembly_adj": assembly_adj,
        "mat_var": mat_var,
        "stack_size": stack_size,
        "maint_diff": maint_diff,
        "rel_motion": rel_motion,
        "compat": compat,
        "W": W_matrix,
        "relation_valid": assembly_adj.copy(),
        "relation_consistent": True,
    }
