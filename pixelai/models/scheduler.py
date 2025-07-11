import torch

from typing import Optional, List, Union


class FlowMatchEulerDiscreteScheduler:
    def __init__(self,
                 num_train_timesteps=1000,
                 shift=1.0):
        
        self.num_train_timesteps = num_train_timesteps

        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)
        timesteps = timesteps[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        # recalculate adjusted timesteps
        self.timesteps = sigmas * num_train_timesteps
        self.sigmas = sigmas.to("cpu")

        self._step_index = None

    def set_timesteps(self,
                      num_inference_steps: Optional[int] = None,
                      sigmas: Optional[torch.Tensor] = None,
                      timesteps: Optional[torch.Tensor] = None):
        
        if sigmas is None and timesteps is None and num_inference_steps is None:
            raise ValueError("At least one of 'sigmas', 'timesteps', or 'num_inference_steps' must be provided.")

        if sigmas is not None and timesteps is not None:
            raise ValueError("Cannot set both 'sigmas' and 'timesteps' at the same time.")

        if num_inference_steps:
            if sigmas is not None and len(sigmas) != num_inference_steps:
                raise ValueError(f"Length of 'sigmas' must match 'num_inference_steps': {len(sigmas)} != {num_inference_steps}.")

            if timesteps is not None and len(timesteps) != num_inference_steps:
                raise ValueError(f"Length of 'timesteps' must match 'num_inference_steps': {len(timesteps)} != {num_inference_steps}.")
        else:
            num_inference_steps = len(sigmas) if sigmas is not None else len(timesteps)

        if sigmas is None:
            if timesteps is None:
                sigmas = torch.linspace(1, 1 / num_inference_steps, num_inference_steps, dtype=torch.float32)
            else:
                sigmas = timesteps / self.num_train_timesteps

        # timestep shifting
        sigmas = self.shift * sigmas / ( 1 + (self.shift - 1) * sigmas)

        if timesteps is None:
            timesteps = sigmas * self.num_train_timesteps

        self.timesteps = timesteps
        self.sigmas = sigmas

    def step(self,
             model_output: torch.Tensor,
             timestep: Union[float, torch.FloatTensor],
             sample: torch.Tensor,
             stochastic_sampling: bool = False):

        pass
        
if __name__ == '__main__':
    scheduler = FlowMatchEulerDiscreteScheduler()