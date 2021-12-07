from omegaconf import DictConfig
import torch
from torch import nn, Tensor


class Embedding(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.pitch_embed = nn.Embedding(num_embeddings=cfg.model.num_pitch,
                                        embedding_dim=cfg.model.d_model)
        self.program_embed = nn.Embedding(num_embeddings=cfg.model.num_program,
                                          embedding_dim=cfg.model.d_model)
        self.velocity_embed = nn.Embedding(
            num_embeddings=cfg.model.num_velocity,
            embedding_dim=cfg.model.d_model)

    def forward(self, pitch: Tensor, program: Tensor, velocity: Tensor):
        pitch_embed = self.pitch_embed(pitch)
        program_embed = self.program_embed(program)
        velocity_embed = self.velocity_embed(velocity)
        return torch.cat([pitch_embed, program_embed, velocity_embed], dim=-1)
