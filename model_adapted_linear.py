import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import weakref


class AdaptedLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, r: int, hypernetwork: nn.Module
    ):
        """
        Linear layer with dynamic LoRA-style adaptation using a hypernetwork.
        Replaces a standard nn.Linear and, if global hypernetwork identifiers
        and layer_id are provided, alters its weight per instance.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            r (int): Low-rank factor.
            hypernetwork (nn.Module): Hypernetwork that predicts adaptation matrices.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self._hypernetwork_ref = weakref.ref(hypernetwork)

        # Base weight and bias parameters.
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        # Initialize parameters.
        self.reset_parameters()

        # Will be set externally by CustomHyperAdapterModel.forward
        self.layer_id: Optional[int] = None
        self.global_HN_ids: Optional[torch.Tensor] = None

    def reset_parameters(self):
        """Initialize weight and bias."""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. If self.global_HN_ids and self.layer_id are set,
        computes dynamic adapter weights via the hypernetwork; otherwise,
        behaves as a standard linear transformation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Dynamic adaptation
        if self.global_HN_ids is not None and self.layer_id is not None:
            HN_ids = self.global_HN_ids  # (batch_size,)
            batch_size = x.size(0)
            # Create per-instance layer_ids tensor
            layer_ids = torch.full_like(HN_ids, fill_value=self.layer_id)
            # Predict adaptation matrices A and B
            hypernetwork = self._hypernetwork_ref()
            A, B = hypernetwork(HN_ids, layer_ids)
            # Reshape into low-rank factors
            A = A.view(batch_size, self.r, self.in_features)  # (batch, r, in_features)
            B = B.view(
                batch_size, self.out_features, self.r
            )  # (batch, out_features, r)
            # Compute delta weight
            delta_weight = torch.bmm(B, A)  # (batch, out_features, in_features)
            # Expand base weight
            base_weight = self.weight.unsqueeze(0).expand(batch_size, -1, -1)
            adapted_weight = base_weight + delta_weight
            # Apply adapted weight
            x_exp = x.view(
                batch_size, self.in_features, x.size(1)
            )  # (batch, in_features, 1)
            out = torch.bmm(adapted_weight, x_exp).squeeze(2)  # (batch, out_features)
            out = out.squeeze(2).permute(0, 2, 1) + self.bias
        else:
            # Fallback to standard linear
            out = F.linear(x, self.weight, self.bias)
        return out
