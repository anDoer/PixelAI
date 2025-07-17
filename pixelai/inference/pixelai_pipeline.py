import torch
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from pathlib import Path

from tqdm import tqdm
from typing import Tuple, Optional, List, Union
from safetensors.torch import load_file

from pixelai.config.default import RuntimeConfig
from pixelai.utils.logging import get_logger
from pixelai.models.pixel_model import PixelTransformer
from pixelai.models.scheduler import FlowMatchEulerDiscreteScheduler

class InferencePipeline:
    def __init__(self, 
                 model: Union[str, PixelTransformer],
                 device = None,
                 weight_dtype = None,
                 **model_kwargs):
        
        self.logger = get_logger(__name__)

        if isinstance(model, PixelTransformer):
            self.transformer = model
        else:
            self.logger.info(f"Loading model from {model}")
            state_dict = load_file(model)

            self.transformer = PixelTransformer(
                patch_size=RuntimeConfig.patch_size if 'patch_size' not in model_kwargs else model_kwargs['patch_size'],
                in_channels=RuntimeConfig.in_channels if 'in_channels' not in model_kwargs else model_kwargs['in_channels'],
                num_layers=RuntimeConfig.num_layers if 'num_layers' not in model_kwargs else model_kwargs['num_layers'],
                attention_head_dim=RuntimeConfig.attention_head_dim if 'attention_head_dim' not in model_kwargs else model_kwargs['attention_head_dim'],
                num_attention_heads=RuntimeConfig.num_attention_heads if 'num_attention_heads' not in model_kwargs else model_kwargs['num_attention_heads'],
                axes_dims_rope=RuntimeConfig.axes_dims_rope if 'axes_dims_rope' not in model_kwargs else model_kwargs['axes_dims_rope'],
            )

            self.transformer.load_state_dict(state_dict)

        if device is None:
            self.transformer.to(device)
        if weight_dtype:
            self.transformer.to(dtype=weight_dtype)

        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=RuntimeConfig.num_train_timesteps,
            shift=RuntimeConfig.scale_shift
        )

        self.logger.info("Inference pipeline initialized with model and scheduler.")


    def run(self,
            image_size: Tuple[int, int] = (24, 28),
            num_samples: int = 4,
            batch_size: int = 4,
            num_inference_steps: int = 50,
            seed: Optional[int] = None,
            output_path: str = 'output/inference/',
            device: Union[str, torch.device] = 'cuda',
            dtype: torch.dtype = torch.float32,
            generator: Optional[torch.Generator] = None) -> List[torch.Tensor]:

        width, height = image_size 

        if generator is None:
            generator = torch.Generator(device=device) 
            if seed:
                generator.manual_seed(seed)

        # generate latents 
        self.logger.info(f"Generating {num_samples} samples with image size {image_size} and batch size {batch_size}")
        num_batches = num_samples // batch_size

        output_images = []

        for b_idx in range(num_batches):
            self.logger.info(f"Processing batch {b_idx + 1}/{num_batches}")

            latents = torch.randn((batch_size, 3, height, width), device=device, dtype=dtype, generator=generator)

            image_ids = PixelTransformer._prepare_latent_image_ids(
                height=height,
                width=width,
                device=latents.device,
                dtype=latents.dtype
            )

            sample = PixelTransformer._pack_latents(latents,
                                                    patch_size=RuntimeConfig.patch_size)

            # adjust the timesteps according to the number of inferecne steps
            # this function considers the schedule shift 
            self.scheduler.set_timesteps(num_inference_steps=num_inference_steps)
            self.scheduler.set_begin_index(0)

            with tqdm(total=len(self.scheduler.timesteps), desc="Inference Progress") as progress_bar:
                for t_idx, timestep in enumerate(self.scheduler.timesteps):
                    with torch.no_grad():
                        timestep = timestep.expand(latents.shape[0]).to(latents.device)

                        model_output = self.transformer(
                            hidden_states=sample,
                            timestep=timestep,
                            img_ids=image_ids,
                        )

                        # update the sample using the scheduler
                        sample = self.scheduler.step(
                            model_output=model_output,
                            timestep=timestep,
                            sample=sample
                        )

                        progress_bar.update(1)
            
            # unpack the latents to images
            images = PixelTransformer._unpack_latents(
                latents=sample,
                height=height,
                width=width,
                patch_size=RuntimeConfig.patch_size
            )

            # Convert images to PIL format
            images = images.to("cpu", dtype=torch.float32)
            images = images.clamp(-1, 1)
            images = (images + 1) / 2.0
            images = images.permute(0, 2, 3, 1)
            images = (images * 255).to(torch.uint8)
            images = images.numpy()
            images = [Image.fromarray(image) for image in images]

            output_images += images

        # Save images to output path
        self.logger.info(f"Saving output images to {output_path}")
        os.makedirs(output_path, exist_ok=True)

        for idx, image in enumerate(output_images):
            image.save(f"{output_path}/image_{idx + 1}.png")

        self.visualize_images(output_images,
                              grid_size=(num_batches, batch_size),
                              save_path=Path(output_path, "visualization.png"))

    def visualize_images(self, 
                         images: List[Image.Image], 
                         grid_size: Tuple[int, int], 
                         upscale_min_side: int = 128, 
                         save_path: Optional[str] = None) -> None:
        """
        Visualizes a list of PIL images in a grid using matplotlib and optionally saves the figure.

        Args:
            images (List[Image.Image]): List of PIL images to visualize.
            grid_size (Tuple[int, int]): Tuple specifying the grid dimensions (rows, cols).
            upscale_min_side (int): Minimum size of the smallest side of the image after upscaling.
            save_path (Optional[str]): Path to save the figure as a PNG image. If None, the figure is not saved.
        """
        rows, cols = grid_size
        assert len(images) <= rows * cols, "Grid size is smaller than the number of images."

        # Upscale images while maintaining aspect ratio
        upscaled_images = []
        for img in images:
            scale_factor = upscale_min_side / min(img.size)
            new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
            upscaled_images.append(img.resize(new_size, Image.Resampling.LANCZOS))

        # Create the grid
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = axes.flatten()

        for ax, img in zip(axes, upscaled_images):
            ax.imshow(img)
            ax.axis('off')

        # Hide unused subplots
        for ax in axes[len(upscaled_images):]:
            ax.axis('off')

        plt.tight_layout()

        # Save the figure if save_path is provided
        if save_path:
            self.logger.info(f"Saving visualization to {save_path}")
            fig.savefig(save_path, format='png', bbox_inches='tight')
