from torch import Tensor, nn


class Embedding(nn.Module):
    def __init__(self, d_model: int, num_token: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=num_token, embedding_dim=d_model, padding_idx=0
        )

    def forward(self, data: Tensor):
        return self.embed(data)
