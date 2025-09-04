import re
from contextlib import contextmanager
import torch
from torch.nn import functional as F
import torch.nn as nn
from transformers import (
    PreTrainedModel,
)
from typing import Any, List, Tuple
from peft import PeftModel, LoraConfig, get_peft_model
from model_hypernetwork import HyperNetworkV2
from model_adapted_linear import AdaptedLinear  # Custom adapter module
from sklearn.preprocessing import LabelEncoder  # For encoding layer numbers


class HyperLoRAModel(PeftModel):
    def __init__(
        self,
        model: torch.nn.Module,
        peft_config: LoraConfig,
        num_embeddings: int = 256,
        loss_weights: dict = None,
        device: torch.device = None,
    ):
        super().__init__(model, peft_config)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.loss = torch.nn.CrossEntropyLoss()
        self.loss_weights = loss_weights

        # collect all the 'query' and 'value' LoRA modules
        self.lora_modules = []
        target_modules = peft_config.target_modules
        for layer in self.model.base_model.roberta.encoder.layer:
            self.lora_modules.extend(
                [getattr(layer.attention.self, module) for module in target_modules]
            )
        # freeze their original LoRA params
        for module in self.lora_modules:
            for p in module.parameters():
                p.requires_grad_(False)

        # hypernetwork dims
        speaker_dim = self.model.config.hidden_size
        hidden_dim = self.model.config.hidden_size
        context_dim = self.model.config.hidden_size
        num_mod = len(self.lora_modules)
        in_dim = self.lora_modules[0].lora_A["default"].weight.shape[1]
        out_dim = self.lora_modules[0].lora_B["default"].weight.shape[0]

        self.hypernet = HyperNetworkV2(
            speaker_dim,
            context_dim,
            hidden_dim,
            in_dim,
            out_dim,
            peft_config.r,
            num_embeddings,
            num_mod,
        )

    @contextmanager
    def _inject_lora_weights(self, A: torch.Tensor, B: torch.Tensor):
        """
        Temporarily override each LoRA module’s forward to use
        F.linear with our generated A/B, instead of its .weight.
        """
        handles = []
        for j, module in enumerate(self.lora_modules):
            # the two Linear submodules created by PEFT
            lora_A = module.lora_A["default"]
            lora_B = module.lora_B["default"]
            wA = A[j].to(self.device)
            wB = B[j].to(self.device)

            # forward-hook replaces the module’s output with F.linear(input, wX, bias)
            handles.append(
                lora_A.register_forward_hook(
                    lambda mod, inp, out, w=wA: F.linear(inp[0], w, mod.bias)
                )
            )
            handles.append(
                lora_B.register_forward_hook(
                    lambda mod, inp, out, w=wB: F.linear(inp[0], w, mod.bias)
                )
            )

        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        num_labels: int = 2,
        num_embeddings: int = 256,
        lora_r: int = 2,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        loss_weights: dict = None,
        device: torch.device = None,
    ) -> "HyperLoRAModel":
        """
        Factory function to create a HyperPeftModel instance from a pretrained model.

        Args:
            pretrained_model_name_or_path (str): Path or identifier for the pretrained model.
            num_labels (int): Number of labels for classification tasks.
            num_embeddings (int): Number of embeddings for the global hypernetwork.
            lora_r (int): Low-rank factor for LoRA.
            lora_alpha (int): Scaling factor for LoRA.
            lora_dropout (float): Dropout rate for LoRA layers.
            device (torch.device): Device to run the model on.
        """
        from transformers import AutoModelForSequenceClassification

        # Load the pretrained model.
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path, num_labels=num_labels
        )

        # Define LoRA configuration.
        peft_config = LoraConfig(
            r=lora_r,
            task_type="SEQ_CLS",
            lora_alpha=lora_alpha,
            # target_modules=["fc1"],  # first MLP weight fc1 (W1)
            target_modules=["query", "value"],
            fan_in_fan_out=False,
            lora_dropout=lora_dropout,
        )

        # Wrap the model with LoRA using PEFT.
        lora_model = get_peft_model(model, peft_config)

        # Instantiate the HyperPeftModel.
        hyper_peft_model = cls(
            lora_model,
            peft_config,
            num_embeddings=num_embeddings,
            loss_weights=loss_weights,
            device=device,
        )
        return hyper_peft_model

    def forward(self, *args, **kwargs):
        # pop off the hypernetwork IDs
        HN_ids = kwargs.pop("annotator_ids").to(self.device)
        batch = HN_ids.size(0)

        # generate all A and B for the whole batch
        A_batch, B_batch = self.hypernet(
            HN_ids
        )  # shapes: (B, M, r, in_dim) and (B, M, out_dim, r)

        logits_list = []
        loss_list = []
        for i in range(batch):
            # for each sample, inject its slice of adapter weights
            Ai = A_batch[i]  # (M, r, in_dim)
            Bi = B_batch[i]  # (M, out_dim, r)

            with self._inject_lora_weights(Ai, Bi):
                # run the model on just this sample
                single_kwargs = {
                    "input_ids": kwargs["input_ids"][i].unsqueeze(0).to(self.device),
                    "attention_mask": kwargs["attention_mask"][i]
                    .unsqueeze(0)
                    .to(self.device),
                    "labels": kwargs["labels"][i].unsqueeze(0).to(self.device),
                }
                out = self.base_model(
                    *args, **single_kwargs
                ).logits  # shape (1, num_labels)
                logits_list.append(out)
                # HN_id = HN_ids[i].unsqueeze(0).item()
                # loss = torch.nn.CrossEntropyLoss(weight=self.loss_weights[HN_id])
                # loss_value = loss(out, single_kwargs["labels"])
                # loss_list.append(loss_value)

        # stack back to (B, num_labels)
        logits = torch.cat(logits_list, dim=0)
        # loss = torch.stack(loss_list, dim=0).mean().to(self.device)
        loss = self.loss(logits, kwargs["labels"].to(self.device))
        # if you want to "catenate" HN_ids into the logits (as before):
        catted = torch.cat([HN_ids.unsqueeze(-1), logits], dim=-1)
        return {"loss": loss, "logits": catted}
