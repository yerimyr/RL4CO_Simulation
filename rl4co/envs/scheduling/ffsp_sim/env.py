import torch
from tensordict.tensordict import TensorDict

from rl4co.envs.scheduling.ffsp.env import FFSPEnv
from .simulator import simulate_makespan


class FFSPSimEnv(FFSPEnv):
    name = "ffsp"

    def __init__(self, simulator_params=None, **kwargs):
        super().__init__(**kwargs)
        self.simulator_params = simulator_params or {}

    def _step(self, td: TensorDict) -> TensorDict:
        td = super()._step(td)

        # FFSPEnv에서는 done 시 reward를 이미 넣지만,
        # 여기서는 시뮬레이터 기반 reward로 덮어씀.
        if td["done"].all():
            schedule = td["schedule"].cpu()
            job_duration = td["job_duration"].cpu()
            makespan = simulate_makespan(
                schedule,
                job_duration,
                **self.simulator_params,
            )
            td.set("reward", -makespan)

        return td
