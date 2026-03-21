import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramNS(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.center = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.context = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        bound = 0.5 / embedding_dim
        nn.init.uniform_(self.center.weight, -bound, bound)
        nn.init.zeros_(self.context.weight)

    def forward(
        self,
        center: torch.Tensor,       # (B,)
        pos_ctx: torch.Tensor,      # (B,)
        neg_ctx: torch.Tensor,      # (B, K)
    ) -> torch.Tensor:
        c = self.center(center)                          # (B, D)
        p = self.context(pos_ctx)                        # (B, D)
        n = self.context(neg_ctx)                        # (B, K, D)

        pos_loss = F.logsigmoid((c * p).sum(dim=1))     # (B,)
        neg_loss = F.logsigmoid(-(n * c.unsqueeze(1)).sum(dim=2)).sum(dim=1)  # (B,)

        return -(pos_loss + neg_loss).mean()

    def get_embeddings(self) -> torch.Tensor:
        return (self.center.weight + self.context.weight) / 2.0
