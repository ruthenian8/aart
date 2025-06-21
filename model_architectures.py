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
