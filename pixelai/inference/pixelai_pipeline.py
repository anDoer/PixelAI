import torch
import argparse

from typing import Tuple, Optional, List, Union
from safetensors.torch import load_file

from pixelai.config.default import RuntimeConfig
from pixelai.utils.logging import get_logger
from pixelai.models.pixel_model import PixelAIModel
from pixelai.models.scheduler import FlowMatchEulerDiscreteScheduler

class InferencePipeline:
    def __init__(self, 
                 model: Union[str, PixelAIModel],
                 device = None,
                 weight_dtype = None,
                 **model_kwargs):
        
        self.logger = get_logger(__name__)

        if isinstance(model, PixelAIModel):
            self.transformer = model
        else:
            self.logger.info(f"Loading model from {model}")
            state_dict = load_file(model)

            self.transformer = PixelAIModel(
                patch_size=RuntimeConfig.patch_size if 'patch_size' not in model_kwargs else model_kwargs['patch_size'],
                in_channels=RuntimeConfig.in_channels if 'in_channels' not in model_kwargs else model_kwargs['in_channels'],
                num_layers=RuntimeConfig.num_layers if 'num_layers' not in model_kwargs else model_kwargs['num_layers'],
                attention_head_dim=RuntimeConfig.attention_head_dim if 'attention_head_dim' not in model_kwargs else model_kwargs['attention_head_dim'],
                num_attention_heads=RuntimeConfig.num_attention_heads if 'num_attention_heads' not in model_kwargs else model_kwargs['num_attention_heads'],
                axes_dims_rope=RuntimeConfig.axes_dims_rope if 'axes_dims_rope' not in model_kwargs else model_kwargs['axes_dims_rope'],
            )

            self.transformer.load_state_dict(state_dict)

        if device:
            self.transformer.to(device)
        if weight_dtype:
            self.transformer.to(dtype=weight_dtype)

        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=RuntimeConfig.num_train_timesteps,
            shift=RuntimeConfig.shift
        )


    def run(self,
            image_size: Tuple[int, int] = (24, 28),
            num_samples: int = 4,
            batch_size: int = 4,
            num_inference_steps: int = 50,
            seed: Optional[int] = None,
            output_path: str = 'output/inference/',
            generator: Optional[torch.Generator] = None) -> List[torch.Tensor]:

        width, height = image_size 

        generator = torch.Generator() 
        if seed:
            generator.manual_seed(seed)

        # generate latents 
        self.logger.info(f"Generating {num_samples} samples with image size {image_size} and batch size {batch_size}")
        num_batches = num_samples // batch_size

        output_images = []

        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps)

        for b_idx in range(num_batches):
            self.logger.info(f"Processing batch {b_idx + 1}/{num_batches}")

            latents = torch.randn((batch_size, 3, height, width))
            packed_latents = PixelAIModel.pack_latents(latents)


