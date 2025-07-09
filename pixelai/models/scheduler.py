import torch
import numpy as np 


class FlowMatchEulerDiscreteScheduler:
    def __init__(self,
                 num_train_timesteps=1000,
                 shift=1.0):
        
        self.num_train_timesteps = num_train_timesteps

        timesteps = np.linspace(0, num_train_timesteps, dtype=np.float32)
        timesteps = timesteps[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        # recalculate adjusted timesteps
        self.timesteps = sigmas * num_train_timesteps
        self.sigmas = sigmas.to("cpu")

        import pdb; pdb.set_trace()

        
if __name__ == '__main__':
    scheduler = FlowMatchEulerDiscreteScheduler()