import torch
import torch.nn as nn 

from typing import Optional
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from diffusers.models.activations import FP32SiLU

class ClassConditionModel(nn.Module):
    def __init__(self,
                 num_classes: int,
                 embedding_dim: int):
        
        super().__init__()
        self.num_classes = num_classes
        self.class_embedding = nn.Embedding(num_classes, embedding_dim)

    def forward(self, class_ids: torch.Tensor) -> torch.Tensor:
        return self.class_embedding(class_ids)


class ConditionProjection(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 hidden_size: int,
                 output_dim: Optional[int] = None,
                 act_fn: str = "gelu_tanh"): 
        super().__init__()

        if output_dim is None:
            output_dim = hidden_size 
            
        self.linear_1 = nn.Linear(in_features=input_dim, out_features=hidden_size, bias=True)
        if act_fn == 'gelu_tanh':
            self.act_1 = nn.GELU(approximate='tanh')
        elif act_fn == 'silu':
            self.act_1 = nn.SiLU()
        elif act_fn == 'silu_fp32':
            self.act_1 = FP32SiLU
        else:
            raise ValueError(f"Unsupported activation function: {act_fn}")

        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=output_dim, bias=True)

    def forward(self, cond_emb: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(cond_emb)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class CombinedTimestepConditionEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.cond_embedder = ConditionProjection(embedding_dim, embedding_dim, act_fn="silu")

    def forward(self, timestep, cond_embeddings):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=cond_embeddings.dtype))  # (N, D)

        cond_projections = self.cond_embedder(cond_embeddings)

        conditioning = timesteps_emb + cond_projections

        return conditioning

class TimestepEmbedder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep):
        timestep_proj = self.time_proj(timestep)
        timestep_emb = self.timestep_embedder(timestep_proj)

        return timestep_emb