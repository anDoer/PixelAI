import torch
import torch.nn as nn 
import torch.nn.functional as F 

from typing import Optional, Tuple, Any, Dict  

from diffusers.models.embeddings import FluxPosEmbed
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.normalization import AdaLayerNormContinuous

from pixelai.models.transformer_blocks import PixelTransformerBlock
from pixelai.models.embeddings import CombinedTimestepConditionEmbeddings, TimestepEmbedder

@maybe_allow_in_graph
class PixelTransformer(nn.Module):

    def __init__(self,
                 patch_size: int = 1,
                 in_channels: int = 3,
                 out_channels: Optional[int] = None,
                 num_layers: int = 32,
                 attention_head_dim: int = 128,
                 num_attention_heads: int = 24,
                 # adjust the axes dims!
                 axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
                 num_classes: Optional[int] = None):
        super().__init__()

        if attention_head_dim != sum(axes_dims_rope):
            raise ValueError(f"Attention head dim {attention_head_dim} must be equal to the sum of axes dims {axes_dims_rope}")

        self.patch_size = patch_size
        self.num_classes = num_classes
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=1000, axes_dim=axes_dims_rope)

        if num_classes is not None:
            self.time_embed = CombinedTimestepConditionEmbeddings(embedding_dim=self.inner_dim)
        else:
            self.time_embed = TimestepEmbedder(embedding_dim=self.inner_dim)

        self.input_embed = nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                PixelTransformerBlock(
                                   dim=self.inner_dim,
                                   num_attention_heads=num_attention_heads,
                                   attention_head_dim=attention_head_dim)
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, 
                                               self.inner_dim, 
                                               elementwise_affine=False, 
                                               eps=1e-6)

        self.proj_out = nn.Linear(self.inner_dim, self.out_channels * patch_size ** 2, bias=True)

    @staticmethod
    def _prepare_latent_image_ids(height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)
    
    @staticmethod
    def _pack_latents(latents: torch.Tensor, 
                      patch_size: int) -> torch.Tensor:

        batch_size, num_channels_latents, height, width = latents.shape

        if patch_size < 1:
            raise ValueError(f"Patch size must be at least 1, got {patch_size}") 
        
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(f"Height {height} and width {width} must be divisible by patch size {patch_size}")

        # latents should be of shape (batch_size, num_channels_latents, height, width)
        latents = latents.view(batch_size, num_channels_latents, height // patch_size, patch_size, width // patch_size, patch_size)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // patch_size) * (width // patch_size), num_channels_latents * patch_size ** 2)

        return latents

    @staticmethod
    def _unpack_latents(latents: torch.Tensor,
                        height: int,
                        width: int,
                        patch_size: int) -> torch.Tensor:
        if patch_size < 1:
            raise ValueError(f"Patch size must be at least 1, got {patch_size}")
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(f"Height {height} and width {width} must be divisible by patch size {patch_size}")

        batch_size, num_patches, channels = latents.shape

        latents = latents.view(batch_size, (height // patch_size), (width // patch_size), channels // (patch_size ** 2), patch_size, patch_size)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (patch_size ** 2), height, width)

        return latents

    def forward(self,
                hidden_states: torch.Tensor,
                timestep: torch.LongTensor,
                img_ids: torch.Tensor,
                conditions: Optional[torch.Tensor] = None,
                joint_attention_kwargs: Optional[Dict[str, Any]] = None,
                ) -> torch.Tensor:
        
        if self.num_classes and conditions is None:
            raise ValueError("Conditions must be provided when num_classes is set")

        hidden_states = self.input_embed(hidden_states)
        timestep = timestep.to(hidden_states.dtype)

        if self.num_classes:
            temb = self.time_embed(timestep, conditions)
        else:
            temb = self.time_embed(timestep)

        image_rotary_emb = self.pos_embed(img_ids) 

        for block in self.transformer_blocks:  
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs
            )
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        return output