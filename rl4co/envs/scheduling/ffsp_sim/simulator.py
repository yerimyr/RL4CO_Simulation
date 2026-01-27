import torch
from typing import Callable, Optional


def simulate_makespan(
    schedule: torch.Tensor,
    job_duration: torch.Tensor,
    *,
    mode: str = "deterministic",
    num_mc: int = 1,
    noise_std: float = 0.1,
    noise_dist: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    seed: Optional[int] = 1234,
):
    """
    schedule: (B, num_machine_total, num_job+1)  start times
    job_duration: (B, num_job+1, num_machine_total)  processing time
    returns: (B,) makespan
    """

    if seed is not None:
        torch.manual_seed(seed)

    # base deterministic end times
    # end_schedule: (B, num_machine_total, num_job+1)
    end_schedule = schedule + job_duration.permute(0, 2, 1)

    # deterministic
    if mode == "deterministic":
        end_time_max, _ = end_schedule[:, :, :-1].max(dim=-1)  # drop dummy job
        end_time_max, _ = end_time_max.max(dim=-1)
        return end_time_max.to(torch.float32)

    # stochastic modes
    total_makespans = []
    for _ in range(num_mc):
        if mode == "gaussian":
            noise = torch.randn_like(end_schedule) * noise_std
            factor = (1.0 + noise).clamp(min=0.0)
        elif mode == "distribution":
            if noise_dist is None:
                raise ValueError("mode='distribution' requires noise_dist callable")
            factor = noise_dist(end_schedule).clamp(min=0.0)
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Choose from ['deterministic', 'gaussian', 'distribution']"
            )

        noisy_end = end_schedule * factor
        end_time_max, _ = noisy_end[:, :, :-1].max(dim=-1)
        end_time_max, _ = end_time_max.max(dim=-1)
        total_makespans.append(end_time_max.to(torch.float32))

    return torch.stack(total_makespans, dim=0).mean(dim=0)
