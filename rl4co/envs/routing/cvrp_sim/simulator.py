# simulator.py 상단에 추가
import multiprocessing as mp
import torch
from typing import Optional, Callable
import os

_POOL = None
_POOL_SIZE = None

def _get_pool(num_workers: int):
    global _POOL, _POOL_SIZE
    if _POOL is None or _POOL_SIZE != num_workers:
        ctx = mp.get_context("spawn")  # Windows 안전
        _POOL = ctx.Pool(processes=num_workers)
        _POOL_SIZE = num_workers
    return _POOL

def _simulate_travel_time_core(locs_ordered, mode, num_mc, noise_std, noise_dist, seed):
    if seed is not None:
        torch.manual_seed(seed)

    diffs = locs_ordered[:, 1:] - locs_ordered[:, :-1]
    distances = torch.norm(diffs, dim=-1)

    if mode == "deterministic":
        return distances.sum(dim=1)

    total_times = []
    for _ in range(num_mc):
        if mode == "gaussian":
            noise = torch.randn_like(distances) * noise_std
            factor = (1.0 + noise).clamp(min=0.0)
        elif mode == "distribution":
            if noise_dist is None:
                raise ValueError("mode='distribution' requires noise_dist callable")
            factor = noise_dist(distances).clamp(min=0.0)
        else:
            raise ValueError(f"Unknown mode: {mode}.")
        travel_time = distances * factor
        total_times.append(travel_time.sum(dim=1))

    return torch.stack(total_times, dim=0).mean(dim=0)

def _worker_cvrp(args):
    
    if not hasattr(_worker_cvrp, "_printed"):
        print(f"[sim worker] pid={os.getpid()}")
        _worker_cvrp._printed = True
        
    (locs_chunk, mode, num_mc, noise_std, noise_dist, seed, wid) = args
    # 워커마다 seed 다르게
    seed = None if seed is None else (seed + wid)
    return _simulate_travel_time_core(
        locs_chunk, mode, num_mc, noise_std, noise_dist, seed
    )

def simulate_travel_time(
    locs_ordered: torch.Tensor,
    *,
    mode: str = "deterministic",
    num_mc: int = 1,
    noise_std: float = 0.1,
    noise_dist=None,
    seed: int | None = 1234,
    num_workers: int = 0,  # 추가
):
    if num_workers is None or num_workers <= 1:
        return _simulate_travel_time_core(
            locs_ordered, mode, num_mc, noise_std, noise_dist, seed
        )

    # 배치 분할
    chunks = torch.chunk(locs_ordered, num_workers, dim=0)
    pool = _get_pool(num_workers)
    tasks = [
        (chunk, mode, num_mc, noise_std, noise_dist, seed, i)
        for i, chunk in enumerate(chunks)
    ]
    results = pool.map(_worker_cvrp, tasks)
    return torch.cat(results, dim=0)
