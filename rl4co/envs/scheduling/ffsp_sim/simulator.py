import torch
import multiprocessing as mp
from typing import Callable, Optional
import os

_POOL = None
_POOL_SIZE = None

def _get_pool(num_workers: int):
    global _POOL, _POOL_SIZE
    if _POOL is None or _POOL_SIZE != num_workers:
        ctx = mp.get_context("spawn")
        _POOL = ctx.Pool(processes=num_workers)
        _POOL_SIZE = num_workers
    return _POOL

def _simulate_makespan_core(schedule, job_duration, mode, num_mc, noise_std, noise_dist, seed):
    if seed is not None:
        torch.manual_seed(seed)

    end_schedule = schedule + job_duration.permute(0, 2, 1)

    if mode == "deterministic":
        end_time_max, _ = end_schedule[:, :, :-1].max(dim=-1)
        end_time_max, _ = end_time_max.max(dim=-1)
        return end_time_max.to(torch.float32)

    total = []
    for _ in range(num_mc):
        if mode == "gaussian":
            noise = torch.randn_like(end_schedule) * noise_std
            factor = (1.0 + noise).clamp(min=0.0)
        elif mode == "distribution":
            if noise_dist is None:
                raise ValueError("mode='distribution' requires noise_dist callable")
            factor = noise_dist(end_schedule).clamp(min=0.0)
        else:
            raise ValueError(f"Unknown mode: {mode}.")
        noisy_end = end_schedule * factor
        end_time_max, _ = noisy_end[:, :, :-1].max(dim=-1)
        end_time_max, _ = end_time_max.max(dim=-1)
        total.append(end_time_max.to(torch.float32))

    return torch.stack(total, dim=0).mean(dim=0)

def _worker_ffsp(args):
    
    if not hasattr(_worker_ffsp, "_printed"):
        print(f"[sim worker] pid={os.getpid()}")
        _worker_ffsp._printed = True
        
    (schedule_chunk, dur_chunk, mode, num_mc, noise_std, noise_dist, seed, wid) = args
    seed = None if seed is None else (seed + wid)
    return _simulate_makespan_core(
        schedule_chunk, dur_chunk, mode, num_mc, noise_std, noise_dist, seed
    )

def simulate_makespan(
    schedule: torch.Tensor,
    job_duration: torch.Tensor,
    *,
    mode: str = "deterministic",
    num_mc: int = 1,
    noise_std: float = 0.1,
    noise_dist: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    seed: Optional[int] = 1234,
    num_workers: int = 0,  # 추가
):
    if num_workers is None or num_workers <= 1:
        return _simulate_makespan_core(
            schedule, job_duration, mode, num_mc, noise_std, noise_dist, seed
        )

    sched_chunks = torch.chunk(schedule, num_workers, dim=0)
    dur_chunks = torch.chunk(job_duration, num_workers, dim=0)

    pool = _get_pool(num_workers)
    tasks = [
        (s, d, mode, num_mc, noise_std, noise_dist, seed, i)
        for i, (s, d) in enumerate(zip(sched_chunks, dur_chunks))
    ]
    results = pool.map(_worker_ffsp, tasks)
    return torch.cat(results, dim=0)
