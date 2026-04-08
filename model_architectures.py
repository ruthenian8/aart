import logging
from contextlib import contextmanager

import torch
from torch.nn import functional as F
from peft import PeftModel, LoraConfig, get_peft_model

from model_hypernetwork import HyperNetworkV2

logger = logging.getLogger(__name__)


class HyperLoRAModel(PeftModel):
    def __init__(
        self,
        model: torch.nn.Module,
        peft_config: LoraConfig,
        num_embeddings: int = 256,
        device: torch.device = None,
    ):
        super().__init__(model, peft_config)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.loss = torch.nn.CrossEntropyLoss()

        # Discover LoRA modules generically via named_modules
        target_modules = set(peft_config.target_modules)
        self.lora_modules = []
        for name, module in self.model.base_model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                # Check if this module's name ends with one of the target module names
                mod_short_name = name.rsplit(".", 1)[-1] if "." in name else name
                if mod_short_name in target_modules:
                    self.lora_modules.append(module)

        if not self.lora_modules:
            raise RuntimeError(
                f"No LoRA modules found for target_modules={peft_config.target_modules}. "
                "Ensure the backbone model is compatible (e.g. RoBERTa-style with query/value projections)."
            )

        logger.info("Discovered %d LoRA modules", len(self.lora_modules))

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
        Temporarily override each LoRA module's forward to use
        F.linear with our generated A/B, instead of its .weight.
        """
        handles = []
        for j, module in enumerate(self.lora_modules):
            lora_A = module.lora_A["default"]
            lora_B = module.lora_B["default"]
            wA = A[j]
            wB = B[j]

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
        device: torch.device = None,
    ) -> "HyperLoRAModel":
        """
        Factory to create a HyperLoRAModel from a pretrained backbone.

        Args:
            pretrained_model_name_or_path: Path or identifier for the pretrained model.
            num_labels: Number of labels for classification tasks.
            num_embeddings: Number of annotator embeddings for the hypernetwork.
            lora_r: Low-rank factor for LoRA.
            lora_alpha: Scaling factor for LoRA.
            lora_dropout: Dropout rate for LoRA layers.
            device: Device to run the model on.
        """
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path, num_labels=num_labels
        )

        peft_config = LoraConfig(
            r=lora_r,
            task_type="SEQ_CLS",
            lora_alpha=lora_alpha,
            target_modules=["query", "value"],
            fan_in_fan_out=False,
            lora_dropout=lora_dropout,
        )

        lora_model = get_peft_model(model, peft_config)

        hyper_peft_model = cls(
            lora_model,
            peft_config,
            num_embeddings=num_embeddings,
            device=device,
        )
        return hyper_peft_model

    def forward(self, *args, **kwargs):
        # pop off the hypernetwork IDs
        HN_ids = kwargs.pop("annotator_ids").to(self.device)
        batch = HN_ids.size(0)
        labels = kwargs.pop("labels", None)

        input_ids = kwargs["input_ids"].to(self.device)
        attention_mask = kwargs["attention_mask"].to(self.device)

        # Group batch items by annotator ID to avoid redundant LoRA weight generation
        unique_ids, inverse_indices = torch.unique(HN_ids, return_inverse=True)

        # Generate LoRA weights for unique annotator IDs only
        A_unique, B_unique = self.hypernet(unique_ids)

        logits = torch.zeros(batch, self.base_model.config.num_labels, device=self.device)

        for uid_idx in range(unique_ids.size(0)):
            # Find all batch indices sharing this annotator ID
            mask = inverse_indices == uid_idx
            batch_indices = mask.nonzero(as_tuple=True)[0]

            Ai = A_unique[uid_idx]  # (M, r, in_dim)
            Bi = B_unique[uid_idx]  # (M, out_dim, r)

            with self._inject_lora_weights(Ai, Bi):
                sub_kwargs = {
                    "input_ids": input_ids[batch_indices],
                    "attention_mask": attention_mask[batch_indices],
                }
                if labels is not None:
                    sub_kwargs["labels"] = labels[batch_indices].to(self.device)

                out = self.base_model(*args, **sub_kwargs).logits
                logits[batch_indices] = out

        # Prepend annotator IDs to logits for metric computation
        catted = torch.cat([HN_ids.unsqueeze(-1), logits], dim=-1)

        result = {"logits": catted}
        if labels is not None:
            loss = self.loss(logits, labels.to(self.device))
            result["loss"] = loss

        return result
