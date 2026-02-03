import multiprocessing as mp
import os
import torch

_POOL = None
_POOL_SIZE = None


def _get_pool(num_workers: int):
    global _POOL, _POOL_SIZE
    if _POOL is None or _POOL_SIZE != num_workers:
        ctx = mp.get_context("spawn")  # Windows
        _POOL = ctx.Pool(processes=num_workers)
        _POOL_SIZE = num_workers
    return _POOL


def _decap_placement_core(pi, probe, raw_pdn, decap, freq, size, num_freq):
    # raw_pdn: [num_freq, size^2, size^2] on CPU
    # decap:   [num_freq, 1, 1] (complex) on CPU
    # freq:    [num_freq] on CPU
    device = raw_pdn.device

    n = m = size
    num_decap = torch.numel(pi)
    z1 = raw_pdn

    decap_flat = decap.reshape(-1)
    z2 = torch.zeros((num_freq, num_decap, num_decap), dtype=torch.float32, device=device)

    qIndx = torch.arange(num_decap, device=device)
    z2[:, qIndx, qIndx] = torch.abs(decap_flat)[:, None].repeat_interleave(
        z2[:, qIndx, qIndx].shape[-1], dim=-1
    )
    pIndx = pi.long()

    aIndx = torch.arange(len(z1[0]), device=device)
    aIndx = torch.tensor(list(set(aIndx.tolist()) - set(pIndx.tolist())), device=device)

    z1aa = z1[:, aIndx, :][:, :, aIndx]
    z1ap = z1[:, aIndx, :][:, :, pIndx]
    z1pa = z1[:, pIndx, :][:, :, aIndx]
    z1pp = z1[:, pIndx, :][:, :, pIndx]
    z2qq = z2[:, qIndx, :][:, :, qIndx]

    zout = z1aa - torch.matmul(torch.matmul(z1ap, torch.inverse(z1pp + z2qq)), z1pa)

    idx = torch.arange(n * m, device=device)
    mask = torch.zeros(n * m, device=device).bool()
    mask[pi] = True
    mask = mask & (idx < probe)
    probe = probe - mask.sum().item()

    zout = zout[:, probe, probe]
    return zout


def _decap_model_core(z_initial, z_final, freq):
    impedance_gap = z_initial - z_final
    reward = torch.sum(impedance_gap * 1000000000 / freq)
    reward = reward / 10
    return reward


def _simulate_decap_reward_core(probe, solution, raw_pdn, decap, freq, size, num_freq):
    # probe: scalar int64
    # solution: 1D tensor of positions
    z_initial = raw_pdn[:, probe, probe].abs()
    z_final = _decap_placement_core(solution, probe, raw_pdn, decap, freq, size, num_freq).abs()
    reward = _decap_model_core(z_initial, z_final, freq)
    return reward


def _worker_dpp(args):
    if not hasattr(_worker_dpp, "_printed"):
        print(f"[sim worker] pid={os.getpid()}")
        _worker_dpp._printed = True

    (probe_chunk, actions_chunk, raw_pdn, decap, freq, size, num_freq) = args
    out = []
    for p, a in zip(probe_chunk, actions_chunk):
        out.append(
            _simulate_decap_reward_core(
                p.item(), a, raw_pdn, decap, freq, size, num_freq
            )
        )
    return torch.stack(out, dim=0)


def simulate_decap_reward(
    probes: torch.Tensor,
    actions: torch.Tensor,
    *,
    raw_pdn: torch.Tensor,
    decap: torch.Tensor,
    freq: torch.Tensor,
    size: int,
    num_freq: int,
    num_workers: int = 0,
):
    """
    probes: [B] on CPU
    actions: [B, K] on CPU
    raw_pdn/decap/freq: on CPU
    returns: [B] reward tensor on CPU
    """
    if num_workers is None or num_workers <= 1:
        out = []
        for p, a in zip(probes, actions):
            out.append(
                _simulate_decap_reward_core(
                    p.item(), a, raw_pdn, decap, freq, size, num_freq
                )
            )
        return torch.stack(out, dim=0)

    chunks = torch.chunk(probes, num_workers, dim=0)
    action_chunks = torch.chunk(actions, num_workers, dim=0)

    pool = _get_pool(num_workers)
    tasks = [
        (p_chunk, a_chunk, raw_pdn, decap, freq, size, num_freq)
        for p_chunk, a_chunk in zip(chunks, action_chunks)
    ]
    results = pool.map(_worker_dpp, tasks)
    return torch.cat(results, dim=0)
