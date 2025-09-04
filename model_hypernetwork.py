import torch
import torch.nn as nn
from typing import Tuple


class HyperNetworkV2(nn.Module):
    def __init__(
        self,
        speaker_dim: int,
        context_dim: int,
        hidden_dim: int,
        in_dim: int,
        out_dim: int,
        r: int,
        num_embeddings: int,
        num_modules: int,
    ):
        super().__init__()
        self.context_emb = nn.Embedding(num_modules, context_dim)
        self.speaker_emb = nn.Embedding(num_embeddings, speaker_dim)
        total_dim = speaker_dim + context_dim
        self.net_A = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, r * in_dim),
        )
        self.net_B = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, out_dim * r),
        )
        # Null-init for delta weights
        nn.init.normal_(self.net_A[-1].weight, std=1e-3)
        nn.init.zeros_(self.net_B[-1].weight)
        nn.init.zeros_(self.net_B[-1].bias)
        self.r, self.in_dim, self.out_dim = r, in_dim, out_dim

    def get_context_embeddings(self, device=None):
        if device is None:
            device = next(self.context_emb.parameters()).device
        idx = torch.arange(self.context_emb.num_embeddings, device=device)
        return self.context_emb(idx)

    def forward(self, HN_ids: torch.Tensor):
        batch = HN_ids.size(0)
        s = self.speaker_emb(HN_ids)  # (batch, speaker_dim)
        c = self.get_context_embeddings(device=HN_ids.device)
        num_mod = c.size(0)
        c_exp = c.unsqueeze(0).expand(batch, num_mod, -1)
        # s_exp = s.expand(batch, num_mod, -1)
        s_exp = s.unsqueeze(1).expand(batch, num_mod, -1)
        inp = torch.cat([s_exp, c_exp], dim=-1)
        flat = inp.view(batch * num_mod, -1)
        A_flat = self.net_A(flat).view(batch, num_mod, self.r, self.in_dim)
        B_flat = self.net_B(flat).view(batch, num_mod, self.out_dim, self.r)
        return A_flat, B_flat
