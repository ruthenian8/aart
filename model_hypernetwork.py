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
