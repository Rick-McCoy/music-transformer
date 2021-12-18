from omegaconf import DictConfig
from torch import nn, Tensor


class Embedding(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(num_embeddings=cfg.model.num_token,
                                  embedding_dim=cfg.model.d_model)

    def forward(self, data: Tensor):
        return self.embed(data)
