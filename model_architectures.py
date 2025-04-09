import re
import torch
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
from dataclasses import dataclass, field
from updated_metrics import losses
from peft import PeftModel, LoraConfig, get_peft_model
from model_hypernetwork import Hypernetwork
from sklearn.preprocessing import LabelEncoder  # For encoding layer numbers


class HyperPeftModel(PeftModel):
    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: LoraConfig,
        num_embeddings: int = 256,
        embedding_dim: int = 128,
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
        self.peft_config = peft_config  # Store for later use.
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
                # Extract layer number with a regex pattern like "layer.<number>"
                match = re.search(r"layer\.(\d+)", name)
                if match:
                    layer_num = int(match.group(1))
                else:
                    layer_num = 0
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
        if HN_ids is not None:
            r = self.peft_config.r
            batch_size = HN_ids.size(0)
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
                a_matrices = A.view(batch_size, a_dim, r)
                b_matrices = B.view(batch_size, b_dim, r)
                delta_weight = torch.bmm(
                    b_matrices, a_matrices
                )  # (batch, b_dim, a_dim)
                base_weight = module.weight.unsqueeze(0).expand(batch_size, -1, -1)
                # Update the module weight with the delta.
                module.weight = nn.Parameter(base_weight + delta_weight)

        # Pass through the base model with the remaining standard inputs.
        new_kwargs = {
            "input_ids": kwargs['input_ids'],
            "attention_mask": kwargs['attention_mask'],
            "labels": kwargs['labels'],
        }
        outputs = self.base_model(*args, **new_kwargs)
        # If the output is not already a dictionary, wrap it; Trainer API expects a dict.
        if not isinstance(outputs, dict):
            outputs = {"logits": outputs}
        return outputs

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
