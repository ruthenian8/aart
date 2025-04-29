import torch
import torch.nn as nn
from typing import Tuple


class Hypernetwork(nn.Module):
    def __init__(
        self,
        num_embeddings: int = 256,
        embedding_dim: int = 128,
        num_layer_embeddings: int = 100,
        layer_embedding_dim: int = 128,
        hidden_dim: int = 128,
        out_A_dim: int = None,
        out_B_dim: int = None,
        dropout: float = 0,
    ):
        """
        Hypernetwork that predicts low-rank adaptation matrices A and B.
        This network integrates two sets of embeddings: one for global features
        and one for adapted layer identifiers.

        Args:
            num_embeddings (int): Number of embeddings for the global feature.
            embedding_dim (int): Dimension of global embeddings.
            num_layer_embeddings (int): Number of embeddings for layer identifiers.
            layer_embedding_dim (int): Dimension of layer embeddings.
            hidden_dim (int): Hidden layer dimension.
            out_A_dim (int): Output dimension for matrix A. Must be divisible by r later.
            out_B_dim (int): Output dimension for matrix B. Must be divisible by r later.
        """
        super(Hypernetwork, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.layer_embedding = nn.Embedding(num_layer_embeddings, layer_embedding_dim)

        # Combined input dimension after concatenating embeddings.
        combined_input_dim = embedding_dim + layer_embedding_dim
        self.fc1 = nn.Linear(combined_input_dim, hidden_dim)
        self.relu = nn.ReLU()

        # If not provided, set default output dimensions.
        if out_A_dim is None:
            out_A_dim = hidden_dim
        if out_B_dim is None:
            out_B_dim = hidden_dim

        self.fc_A = nn.Linear(hidden_dim, out_A_dim)
        self.fc_B = nn.Linear(hidden_dim, out_B_dim)

        # Initialize fc_B so that initially ΔW = 0.
        nn.init.constant_(self.fc_B.weight, 0.0)
        nn.init.constant_(self.fc_B.bias, 0.0)
        # Initialize fc_A with small random weights.
        nn.init.normal_(self.fc_A.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_A.bias, 0.0)

    def forward(
        self, HN_ids: torch.Tensor, layer_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the hypernetwork.

        Args:
            HN_ids (torch.Tensor): Tensor with global hypernetwork IDs.
            layer_ids (torch.Tensor): Tensor with adapted layer identifiers.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted matrices A and B.
        """
        emb_global = self.embedding(HN_ids)  # (batch, embedding_dim)
        emb_layer = self.layer_embedding(layer_ids)  # (batch, layer_embedding_dim)
        # Concatenate the embeddings.
        combined_emb = torch.cat(
            [emb_global, emb_layer], dim=-1
        )  # (batch, embedding_dim + layer_embedding_dim)
        hidden = self.relu(self.fc1(combined_emb))
        hidden = self.dropout(hidden)
        A = self.fc_A(hidden)
        B = self.fc_B(hidden)
        return A, B


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
            nn.ReLU(),
            nn.Linear(hidden_dim, r * in_dim),
        )
        self.net_B = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
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
