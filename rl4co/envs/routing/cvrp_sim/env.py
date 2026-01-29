import torch
from tensordict.tensordict import TensorDict

from rl4co.envs.routing.cvrp.env import CVRPEnv
from rl4co.utils.ops import gather_by_index  
from .simulator import simulate_travel_time  


class CVRPSimEnv(CVRPEnv):  

    name = "cvrp"

    def __init__(self, simulator_params=None, **kwargs):
        super().__init__(**kwargs)
        self.simulator_params = simulator_params or {}

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:  
        locs_ordered = torch.cat(
            [
                td["locs"][..., 0:1, :],      
                gather_by_index(td["locs"], actions),  
            ],
            dim=1,
        )  

        total_time = simulate_travel_time(
            locs_ordered.cpu(),
            **self.simulator_params,
        )

        return -total_time.to(td.device)