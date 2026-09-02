import logging
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


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
        speaker_emb: Optional[nn.Embedding] = None,
    ):
        super().__init__()
        self.context_emb = nn.Embedding(num_modules, context_dim)
        if speaker_emb is None:
            self.speaker_emb = nn.Embedding(num_embeddings, speaker_dim)
        else:
            # Keep one annotator embedding table when several shape-specific
            # hypernetworks are composed together.
            self.speaker_emb = speaker_emb

        # Register buffer for module indices to avoid repeated allocation
        self.register_buffer(
            "module_indices",
            torch.arange(num_modules),
            persistent=False,
        )

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
        self.num_modules = num_modules

    def get_context_embeddings(self, device=None):
        idx = self.module_indices
        if device is not None and idx.device != device:
            idx = idx.to(device)
        return self.context_emb(idx)

    def forward(self, HN_ids: torch.Tensor):
        batch = HN_ids.size(0)
        s = self.speaker_emb(HN_ids)  # (batch, speaker_dim)
        c = self.get_context_embeddings(device=HN_ids.device)
        num_mod = c.size(0)
        c_exp = c.unsqueeze(0).expand(batch, num_mod, -1)
        s_exp = s.unsqueeze(1).expand(batch, num_mod, -1)
        inp = torch.cat([s_exp, c_exp], dim=-1)
        flat = inp.view(batch * num_mod, -1)
        A_flat = self.net_A(flat).view(batch, num_mod, self.r, self.in_dim)
        B_flat = self.net_B(flat).view(batch, num_mod, self.out_dim, self.r)

        # Shape assertions for debugging (disabled when running with python -O)
        if __debug__:
            assert A_flat.shape == (batch, num_mod, self.r, self.in_dim), (
                f"Expected A shape ({batch}, {num_mod}, {self.r}, {self.in_dim}), got {A_flat.shape}"
            )
            assert B_flat.shape == (batch, num_mod, self.out_dim, self.r), (
                f"Expected B shape ({batch}, {num_mod}, {self.out_dim}, {self.r}), got {B_flat.shape}"
            )

        return A_flat, B_flat


class HyperNetworkCollection(nn.Module):
    """Shape-specific hypernetworks sharing one annotator embedding table."""

    def __init__(
        self,
        speaker_dim: int,
        context_dim: int,
        hidden_dim: int,
        r: int,
        num_embeddings: int,
        group_specs: Sequence[Tuple[int, int, int]],
    ):
        super().__init__()
        self.speaker_emb = nn.Embedding(num_embeddings, speaker_dim)
        self.hypernets = nn.ModuleList(
            HyperNetworkV2(
                speaker_dim,
                context_dim,
                hidden_dim,
                in_dim,
                out_dim,
                r,
                num_embeddings,
                num_modules,
                speaker_emb=self.speaker_emb,
            )
            for in_dim, out_dim, num_modules in group_specs
        )

    def forward(self, HN_ids: torch.Tensor):
        return [hypernet(HN_ids) for hypernet in self.hypernets]
