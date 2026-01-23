import torch
from typing import Callable, Optional


def simulate_travel_time(
    locs_ordered: torch.Tensor,
    *,
    mode: str = "deterministic",
    num_mc: int = 1,
    noise_std: float = 0.1,
    noise_dist: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    seed: Optional[int] = 1234,
):
    """
    Parameters
    ----------
    locs_ordered : torch.Tensor
        Shape: (B, T, 2)
        B: number of routes (batch size)
        T: route length (including depot)
        2: 2D coordinates (x, y)

    mode : str
        - "deterministic" : pure Euclidean distance (no randomness)
        - "gaussian"      : Gaussian noise on distances
        - "distribution"  : user-defined noise distribution

    num_mc : int
        Number of Monte Carlo samples per route (used only in stochastic modes)

    noise_std : float
        Standard deviation for Gaussian noise (mode="gaussian")

    noise_dist : Callable, optional
        Function that takes `distances` tensor and returns multiplicative noise
        Example: lambda d: torch.distributions.LogNormal(0, 0.2).sample(d.shape)

    seed : int or None
        Random seed for reproducibility (None → no seed fixing)

    Returns
    -------
    torch.Tensor
        Shape: (B,)
        Mean total travel time per route
    """

    if seed is not None:
        torch.manual_seed(seed)

    B, T, _ = locs_ordered.shape

    # ----------------------------------------------------
    # Deterministic base distances (always computed)
    # ----------------------------------------------------
    diffs = locs_ordered[:, 1:] - locs_ordered[:, :-1]   # (B, T-1, 2)
    distances = torch.norm(diffs, dim=-1)                # (B, T-1)

    # ----------------------------------------------------
    # Deterministic mode (validation / oracle evaluation)
    # ----------------------------------------------------
    if mode == "deterministic":
        return distances.sum(dim=1)

    # ----------------------------------------------------
    # Stochastic modes
    # ----------------------------------------------------
    total_times = []

    for _ in range(num_mc):

        if mode == "gaussian":
            # multiplicative Gaussian noise
            noise = torch.randn_like(distances) * noise_std
            factor = (1.0 + noise).clamp(min=0.0)

        elif mode == "distribution":
            if noise_dist is None:
                raise ValueError(
                    "mode='distribution' requires noise_dist callable"
                )
            factor = noise_dist(distances)
            factor = factor.clamp(min=0.0)

        else:
            raise ValueError(
                f"Unknown mode: {mode}. "
                "Choose from ['deterministic', 'gaussian', 'distribution']"
            )

        travel_time = distances * factor
        total_times.append(travel_time.sum(dim=1))  # (B,)

    # Monte Carlo average
    return torch.stack(total_times, dim=0).mean(dim=0)
