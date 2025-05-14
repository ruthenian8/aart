import re
from contextlib import contextmanager
import torch
from torch.nn import functional as F
import torch.nn as nn
from transformers import (
    AutoConfig,
    PreTrainedModel,
    RobertaModel,
    RobertaForSequenceClassification,
    BertModel,
    BertConfig,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from updated_metrics import losses
from peft import PeftModel, LoraConfig, get_peft_model
from model_hypernetwork import Hypernetwork, HyperNetworkV2
from model_adapted_linear import AdaptedLinear  # Custom adapter module
from sklearn.preprocessing import LabelEncoder  # For encoding layer numbers


class CustomHyperAdapterModel(nn.Module):
    def __init__(
        self,
        base_model: "PreTrainedModel",
        adapter_config: dict,
        num_embeddings: int = 256,
        embedding_dim: int = 768,
        hidden_dim: int = 128,
        num_layer_embeddings: int = 100,
        layer_embedding_dim: int = 128,
        r: int = 2,
        out_dim=768,
    ):
        """
        Custom model that integrates a hypernetwork to predict adapter weights
        from global HN_ids and layer_ids, and uses custom adapter modules (e.g. AdaptedLinear)
        to replace target layers in the base model.

        Args:
            base_model (PreTrainedModel): The pretrained base model.
            adapter_config (dict): A configuration dictionary for adapter parameters (e.g. low-rank factor).
            num_embeddings (int): Number of embeddings for global hypernetwork identifiers.
            embedding_dim (int): Dimensionality for global hypernetwork embeddings.
            hidden_dim (int): Hidden dimension of the hypernetwork.
            num_layer_embeddings (int): Number of distinct embeddings for layer identifiers.
            layer_embedding_dim (int): Embedding dimension for layer identifiers.
            r (int): The low-rank factor used by the adapters.
        """
        super(CustomHyperAdapterModel, self).__init__()
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.adapter_config = adapter_config
        self.r = r
        # Create the hypernetwork that will predict low-rank adapter weights.
        self.hypernetwork = Hypernetwork(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            num_layer_embeddings=num_layer_embeddings,
            layer_embedding_dim=layer_embedding_dim,
            hidden_dim=hidden_dim,
            out_A_dim=r * embedding_dim,
            out_B_dim=r * out_dim,
        )
        # Replace target layers in the base model with our custom adapter modules.
        self._replace_adapter_layers()

    def _replace_adapter_layers(self):
        """
        Replace selected layers in the base model with custom adapter modules.
        In this example, we assume the base model is roberta-like and replace the
        feed-forward “intermediate” dense layer in each transformer block with an AdaptedLinear.
        The replaced module stores its layer identifier in its attribute "layer_id".
        """
        if hasattr(self.base_model, "roberta") and hasattr(
            self.base_model.roberta, "encoder"
        ):
            for layer_idx, block in enumerate(self.base_model.roberta.encoder.layer):
                # Retrieve the original dense layer from the intermediate feed-forward block.
                original_dense = block.intermediate.dense
                in_features = original_dense.in_features
                out_features = original_dense.out_features
                # Create a new custom adapter module that uses AdaptedLinear.
                adapted_linear = AdaptedLinear(
                    in_features, out_features, self.r, self.hypernetwork
                )
                # Save the layer index (will be used as layer_id during hypernetwork inference).
                adapted_linear.layer_id = layer_idx  # Stored as an integer
                # Replace the original dense layer with our adapted module.
                block.intermediate.dense = adapted_linear
        else:
            raise ValueError(
                "Base model architecture not recognized for adapter replacement."
            )

    def forward(
        self, input_ids, attention_mask=None, labels=None, HN_ids=None, **kwargs
    ):
        """
        The forward pass for the CustomHyperAdapterModel.
        If HN_ids (global hypernetwork IDs) is provided, it traverses the base model to
        assign these identifiers to each custom adapter module. In turn, each adapted module
        will use its stored layer_id to call the hypernetwork and generate dynamic adapter weights.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Labels for training.
            HN_ids (torch.Tensor): Global hypernetwork indices (e.g. from a LabelEncoder).
            **kwargs: Additional arguments for the base model.

        Returns:
            A dict containing outputs from the base model (e.g. loss and logits).
        """
        HN_ids: torch.Tensor = kwargs.pop("annotator_ids", None)
        # Traverse the base model and inject HN_ids into custom adapter modules.
        # Each custom adapter (AdaptedLinear) is expected to use its stored layer_id and the provided HN_ids.
        if HN_ids is not None:

            def inject_HN_ids(module):
                for child in module.children():
                    # If this is an adapted module and it has a stored layer_id, assign HN_ids.
                    if isinstance(child, AdaptedLinear) and hasattr(child, "layer_id"):
                        child.global_HN_ids = (
                            HN_ids  # Save HN_ids as an attribute for use in forward
                        )
                    inject_HN_ids(child)

            inject_HN_ids(self.base_model)

        # Forward the remaining inputs to the base model.
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        if not isinstance(outputs, dict):
            outputs = {
                "loss": outputs.loss,
                "logits": outputs.logits,
                **{k: v for k, v in outputs.items() if k not in ("loss", "logits")},
            }
        return outputs

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        num_labels: int = 2,
        num_embeddings: int = 256,
        embedding_dim: int = 768,
        hidden_dim: int = 128,
        num_layer_embeddings: int = 100,
        layer_embedding_dim: int = 128,
        r: int = 2,
        out_dim: int = 768,
        **kwargs,
    ) -> "CustomHyperAdapterModel":
        """
        Factory method that loads a pretrained model and returns an instance
        of CustomHyperAdapterModel with custom adapter modules.

        Args:
            pretrained_model_name_or_path (str): The identifier or path of the pretrained model.
            num_labels (int): Number of labels (for classification tasks).
            num_embeddings (int): Number of embeddings for global hypernetwork identifiers.
            embedding_dim (int): Dimensionality for global hypernetwork embeddings.
            hidden_dim (int): Hidden layer dimension of the hypernetwork.
            num_layer_embeddings (int): Number of embeddings for layer identifiers.
            layer_embedding_dim (int): Embedding dimension for layer identifiers.
            r (int): Low-rank factor for the adapters.
            **kwargs: Additional keyword arguments for loading the pretrained model.

        Returns:
            An instance of CustomHyperAdapterModel.
        """
        from transformers import AutoModelForSequenceClassification

        base_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path, num_labels=num_labels
        )
        # Minimal adapter configuration; extend as needed.
        adapter_config = {"r": r}
        return cls(
            base_model,
            adapter_config,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layer_embeddings=num_layer_embeddings,
            layer_embedding_dim=layer_embedding_dim,
            r=r,
            out_dim=out_dim,
        )


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


class HyperPeftModel(PeftModel):
    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: LoraConfig,
        num_embeddings: int = 256,
        embedding_dim: int = 768,
        hidden_dim: int = 128,
        num_layer_embeddings: int = 100,
        layer_embedding_dim: int = 128,
    ):
        """
        Combined model integrating a hypernetwork with a PEFT-based model.
        Each LoRA adapter in the base model is updated distinctly based on the
        layer number extracted from its module name, after encoding with LabelEncoder.

        Args:
            model (PreTrainedModel): Base pretrained model.
            peft_config (LoraConfig): Configuration for the LoRA adapter.
            num_embeddings (int): Number of embeddings for the global hypernetwork.
            embedding_dim (int): Dimension of the global hypernetwork embeddings.
            hidden_dim (int): Hidden layer dimension in the hypernetwork.
            num_layer_embeddings (int): Number of adapted layer identifiers.
            layer_embedding_dim (int): Dimension of adapted layer embeddings.
        """
        super(HyperPeftModel, self).__init__(model, peft_config)
        self.config = model.config
        # Create the global hypernetwork.
        self.hypernetwork = Hypernetwork(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            num_layer_embeddings=num_layer_embeddings,
            layer_embedding_dim=layer_embedding_dim,
            hidden_dim=hidden_dim,
            # Assuming each LoRA module expects delta parameters sized by (r * embedding_dim)
            out_A_dim=peft_config.r * embedding_dim,
            out_B_dim=peft_config.r * embedding_dim,
        )
        self.loss = torch.nn.CrossEntropyLoss()
        self._patch_lora_modules()

    def _patch_lora_modules(self) -> None:
        """
        Identify and store references to modules in the base model that have LoRA parameters.
        Instead of using a dedicated adapter_id, we extract a layer number from the module's
        name using a regex, then encode these numbers with a LabelEncoder.
        """
        modules_with_layer: List[Tuple[nn.Module, int]] = []
        for name, module in self.base_model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                for param in module.lora_A.parameters():
                    param.requires_grad = False
                for param in module.lora_B.parameters():
                    param.requires_grad = False
                # Extract layer number with a regex pattern like "layer.<number>"
                match = re.search(r"layer\.(\d+)", name)
                layer_num = int(match.group(1)) if match else 0
                modules_with_layer.append((module, layer_num))

        # Fit a LabelEncoder on the extracted layer numbers.
        layer_numbers = [layer_num for (_, layer_num) in modules_with_layer]
        le = LabelEncoder()
        encoded_layer_numbers = le.fit_transform(layer_numbers)
        # Save the mapping (if needed later) and store the encoded layer id.
        self.layer_label_encoder = le

        # Save modules with their encoded layer id.
        self.lora_modules: List[Tuple[nn.Module, int]] = []
        for (module, _), encoded_layer in zip(
            modules_with_layer, encoded_layer_numbers
        ):
            self.lora_modules.append((module, int(encoded_layer)))

    @contextmanager
    def _patch_modules_temporarily(
        self, effective_weights: List[Tuple[nn.Module, torch.Tensor]]
    ):
        """Temporarily patch modules’ weights for the forward pass."""
        old_weights = {}
        for module, new_weight in effective_weights:
            old_weights[module] = module.weight
            # module.weight = new_weight
            module._parameters["weight"] = new_weight
        try:
            yield
        finally:
            for module, _ in effective_weights:
                # module.weight = old_weights[module]
                module._parameters["weight"] = old_weights[module]

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Forward pass of the HyperPeftModel.
        If 'HN_ids' is provided in kwargs, the hypernetwork predicts delta matrices
        that are applied to each LoRA module. The underlying model is then invoked with
        the remaining inputs, making the code compatible with the Trainer API.

        Expected inputs include standard keys like 'input_ids', 'attention_mask', 'labels',
        etc. Additionally, the batch must contain "annotator_ids" (output from LabelEncoder) which
        is used as the global hypernetwork input.

        Args:
            *args: Positional arguments for the base model.
            **kwargs: Keyword arguments for the base model (including "annotator_ids").

        Returns:
            The output of the base model, typically a dictionary with "loss" and "logits".
        """
        HN_ids: torch.Tensor = kwargs.pop("annotator_ids", None)
        new_kwargs = {
            "input_ids": kwargs["input_ids"],
            "attention_mask": kwargs["attention_mask"],
            "labels": kwargs["labels"],
        }
        if HN_ids is not None:
            r = self.peft_config["default"].r
            batch_size = HN_ids.size(0)
            patch_list = []
            for module, encoded_layer in self.lora_modules:
                # Create a tensor for the layer identifier that matches the batch size.
                layer_ids = torch.full_like(HN_ids, fill_value=encoded_layer)
                # Predict adaptation matrices using both the global HN_ids and the layer_ids.
                A, B = self.hypernetwork(HN_ids, layer_ids)
                # Reshape outputs into low-rank factors.
                a_dim = (
                    self.hypernetwork.fc_A.out_features // r
                )  # Typically equals embedding_dim.
                b_dim = (
                    self.hypernetwork.fc_B.out_features // r
                )  # Typically equals embedding_dim.
                a_matrices = A.view(batch_size, r, a_dim)
                b_matrices = B.view(batch_size, b_dim, r)
                delta_weight = torch.bmm(
                    b_matrices, a_matrices
                )  # (batch, b_dim, a_dim)
                base_weight = (
                    module.weight.detach().unsqueeze(0).expand(batch_size, -1, -1)
                )
                # base_weight = module.weight.unsqueeze(0).expand(batch_size, -1, -1)
                # Update the module weight with the delta.
                # module._parameters["weight"] = nn.Parameter(base_weight + delta_weight)
                patch_list.append((module, base_weight + delta_weight))
            with self._patch_modules_temporarily(patch_list):
                outputs = self.base_model(*args, **new_kwargs)
        else:
            outputs = self.base_model(*args, **new_kwargs)

        # outputs = self.base_model(*args, **new_kwargs)

        # If the output is not already a dictionary, wrap it; Trainer API expects a dict.
        loss = self.loss(outputs.logits, new_kwargs["labels"])
        if not isinstance(outputs, dict):
            outputs = {**outputs, "loss": loss, "logits": outputs.logits}

        catted_logits = torch.cat((HN_ids.reshape(-1, 1), outputs["logits"]), 1)
        return {**outputs, "loss": loss, "logits": catted_logits}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        num_labels: int = 2,
        num_embeddings: int = 256,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layer_embeddings: int = 100,
        layer_embedding_dim: int = 128,
        lora_r: int = 2,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ) -> "HyperPeftModel":
        """
        Factory function to create a HyperPeftModel instance from a pretrained model.

        Args:
            pretrained_model_name_or_path (str): Path or identifier for the pretrained model.
            num_embeddings (int): Number of embeddings for the global hypernetwork.
            embedding_dim (int): Dimension of the global hypernetwork embeddings.
            hidden_dim (int): Hidden layer dimension in the hypernetwork.
            num_layer_embeddings (int): Number of adapted layer identifiers.
            layer_embedding_dim (int): Dimension of adapted layer embeddings.
            lora_r (int): Low-rank factor.
            lora_alpha (int): Scaling factor for LoRA.
            lora_dropout (float): Dropout rate for LoRA layers.

        Returns:
            HyperPeftModel: An instance of the combined hypernetwork and PEFT model.
        """
        from transformers import AutoModelForSequenceClassification

        # Load the pretrained model.
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path, num_labels=num_labels
        )

        # Define LoRA configuration.
        peft_config = LoraConfig(
            task_type="SEQ_CLS",  # Adjust based on your task.
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        # Wrap the model with LoRA using PEFT.
        lora_model = get_peft_model(model, peft_config)

        # Instantiate the HyperPeftModel.
        hyper_peft_model = cls(
            lora_model,
            peft_config,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layer_embeddings=num_layer_embeddings,
            layer_embedding_dim=layer_embedding_dim,
        )
        return hyper_peft_model


class MultiTaskClassifier(RobertaForSequenceClassification):
    def __init__(self, config, balancing_weights, task_labels=["majority"]):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # might need to rename this depending on the model
        self.roberta = RobertaModel(config)
        nhid = self.roberta.config.hidden_size
        print("@@@ nhid: ", nhid)

        self.task_labels = task_labels
        self.linear_layer = dict()
        for task in task_labels:
            self.linear_layer[task] = nn.Linear(nhid, self.num_labels).to(
                torch.device("cuda")
            )
        self.balancing_weights = balancing_weights
        self.create_loss_functions()
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # output will be (batch_size, seq_length, hidden_size)
        hidden = outputs.last_hidden_state[:, 0, :]

        logits = dict()
        for task in self.task_labels:
            logits[task] = self.linear_layer[task](hidden)

        # predictions = {task: [x.item() for x in torch.argmax(logits[task], dim=-1)] for task in self.task_labels}
        labels = {k: kwargs[k] for k in kwargs.keys() if k in self.task_labels}
        loss = self.calculate_loss(labels=labels, logits=logits)

        # logits = torch.cat([logits_tensor for logits_tensor in logits.values()])

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def calculate_loss(self, labels, logits):
        task_loss_dict = dict()

        for task_label in self.task_labels:
            if (labels[task_label] == -1).all().item():
                continue
            if labels[task_label].isnan().all().item():
                continue
            task_loss_dict[task_label] = self.losses[task_label](
                logits[task_label][
                    ~torch.any(labels[task_label].isnan().view(-1, 1), dim=1)
                ],
                target=labels[task_label][
                    ~torch.any(labels[task_label].isnan().view(-1, 1), dim=1)
                ],
            )
            # (logits[task_label], target=labels[task_label])

        total_loss = sum(task_loss_dict.values())
        return total_loss

    def create_loss_functions(self):
        self.losses = dict()

        for task_label in self.task_labels:
            self.losses[task_label] = nn.CrossEntropyLoss(
                weight=self.balancing_weights[task_label], ignore_index=-1
            )


@dataclass
class AARTSequenceClassifierOutput(SequenceClassifierOutput):
    ce_loss: Optional[torch.FloatTensor] = None
    l2_norm: Optional[torch.FloatTensor] = None
    contrastive_loss: Optional[torch.FloatTensor] = None


class AARTClassifier(RobertaForSequenceClassification):

    def __init__(self, config, label_weights, annotator_weights=[], embd_type_cnt={}):
        super().__init__(config)
        self.config = config
        self.roberta = RobertaModel(config)
        nhid = self.roberta.config.hidden_size
        print("@@@ nhid: ", nhid)
        print("@@ num labels:", config.num_labels)
        self.emb_names = list(embd_type_cnt.keys())
        for k, cnt in embd_type_cnt.items():
            rand_weight = torch.rand(size=(cnt, nhid))
            setattr(
                self,
                f"{k}_embeddings",
                nn.Embedding.from_pretrained(rand_weight, freeze=False).to(
                    torch.device("cuda")
                ),
            )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(nhid, config.num_labels)
        self.label_balancing_weights = label_weights
        if not embd_type_cnt:
            self.annotator_balancing_weights = []
        else:
            self.annotator_balancing_weights = annotator_weights

        # Initialize weights and apply final processing
        self.post_init()

    def calculate_loss(self, labels, logits, text_ids, other_args):
        # elif self.config.problem_type == "single_label_classification":
        if len(self.annotator_balancing_weights):
            loss_fct = nn.CrossEntropyLoss(
                weight=self.label_balancing_weights, ignore_index=-1, reduction="none"
            )
            classification_loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1)
            )
            classification_loss = (
                classification_loss
                * self.annotator_balancing_weights[other_args[f"annotator_ids"]]
            ).sum() / self.annotator_balancing_weights[
                other_args[f"annotator_ids"]
            ].sum()
        else:
            loss_fct = nn.CrossEntropyLoss(
                weight=self.label_balancing_weights, ignore_index=-1
            )
            classification_loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1)
            )

        if self.emb_names:
            contrastive_loss_funct = losses.ContrastiveLoss()  # losses.NTXentLoss()
            l2_norm = torch.tensor(0.0, requires_grad=True)
            contrastive_loss = torch.tensor(0.0, requires_grad=True)

            for k in self.emb_names:
                l2_norm = (
                    l2_norm
                    + torch.linalg.vector_norm(
                        getattr(self, f"{k}_embeddings").weight, dim=1, ord=2
                    ).mean()
                )
                # todo what will happen to the the same embeddings? for example a0 and a0? or hispanic and hispanic?
                contrastive_loss = contrastive_loss + contrastive_loss_funct(
                    getattr(self, f"{k}_embeddings")(other_args[f"{k}_ids"]),
                    labels=labels.view(-1),
                    mask_labels=text_ids,
                )

        return classification_loss, l2_norm, contrastive_loss

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        cls_embeddings = self.dropout(cls_embeddings)
        batch_embeddings = cls_embeddings
        for k in self.emb_names:
            batch_embeddings = batch_embeddings + getattr(self, f"{k}_embeddings")(
                kwargs[f"{k}_ids"]
            )

        batch_embeddings = self.LayerNorm(batch_embeddings)
        # batch_embeddings = self.dropout(batch_embeddings)
        logits = self.classifier(batch_embeddings)

        classification_loss, l2_norm, contrastive_loss = self.calculate_loss(
            logits=logits, labels=labels, text_ids=kwargs["text_ids"], other_args=kwargs
        )

        logits = torch.cat((kwargs[f"annotator_ids"].reshape(-1, 1), logits), 1)

        return AARTSequenceClassifierOutput(
            ce_loss=classification_loss,
            l2_norm=l2_norm,
            contrastive_loss=contrastive_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
