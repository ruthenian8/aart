import copy
import inspect
import logging
from contextlib import contextmanager
from typing import Optional, Sequence

import torch
from torch.nn import functional as F
from peft import PeftModel, LoraConfig, get_peft_model

from model_hypernetwork import HyperNetworkCollection

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
        self.task_type = str(peft_config.task_type)

        # Discover LoRA modules generically via named_modules
        target_modules = set(peft_config.target_modules or [])
        self.lora_modules = []
        for name, module in self.model.base_model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                mod_short_name = name.rsplit(".", 1)[-1] if "." in name else name
                if not target_modules or mod_short_name in target_modules:
                    self.lora_modules.append(module)

        if not self.lora_modules:
            raise RuntimeError(
                f"No LoRA modules found for target_modules={peft_config.target_modules}. "
                "Ensure the backbone model and target modules are LoRA-compatible."
            )

        module_groups = {}
        for module in self.lora_modules:
            shape = (
                module.lora_A["default"].weight.shape[1],
                module.lora_B["default"].weight.shape[0],
            )
            module_groups.setdefault(shape, []).append(module)
        self.lora_module_groups = list(module_groups.values())

        logger.info("Discovered %d LoRA modules", len(self.lora_modules))

        # freeze their original LoRA params
        for module in self.lora_modules:
            for p in module.parameters():
                p.requires_grad_(False)

        # hypernetwork dims
        speaker_dim = self.model.config.hidden_size
        hidden_dim = self.model.config.hidden_size
        context_dim = self.model.config.hidden_size
        group_specs = [
            (in_dim, out_dim, len(modules))
            for (in_dim, out_dim), modules in module_groups.items()
        ]
        self.hypernet = HyperNetworkCollection(
            speaker_dim,
            context_dim,
            hidden_dim,
            peft_config.r,
            num_embeddings,
            group_specs,
        )

    @contextmanager
    def _inject_lora_weights(self, weights):
        """
        Temporarily override each LoRA module's forward to use
        F.linear with our generated A/B, instead of its .weight.
        """
        handles = []
        for (modules, (A_group, B_group)) in zip(
            self.lora_module_groups, weights
        ):
            for j, module in enumerate(modules):
                lora_A = module.lora_A["default"]
                lora_B = module.lora_B["default"]
                wA = A_group[j]
                wB = B_group[j]

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
        task_type: str = "SEQ_CLS",
        target_modules: Optional[Sequence[str]] = None,
        fan_in_fan_out: bool = False,
    ) -> "HyperLoRAModel":
        """
        Factory to create a HyperLoRAModel from a pretrained backbone.

        Args:
            pretrained_model_name_or_path: Path or identifier for the pretrained model.
            num_labels: Number of labels for sequence-classification tasks.
            num_embeddings: Number of annotator embeddings for the hypernetwork.
            lora_r: Low-rank factor for LoRA.
            lora_alpha: Scaling factor for LoRA.
            lora_dropout: Dropout rate for LoRA layers.
            device: Device to run the model on.
            task_type: PEFT task type, either ``SEQ_CLS`` or ``CAUSAL_LM``.
            target_modules: Optional projection names to adapt. When omitted,
                PEFT selects the defaults for the backbone architecture.
            fan_in_fan_out: Whether target linear weights use fan-in/fan-out layout.
        """
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

        task_type = task_type.upper()
        if task_type == "SEQ_CLS":
            model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path, num_labels=num_labels
            )
        elif task_type == "CAUSAL_LM":
            model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)
        else:
            raise ValueError("task_type must be either 'SEQ_CLS' or 'CAUSAL_LM'")

        peft_config = LoraConfig(
            r=lora_r,
            task_type=task_type,
            lora_alpha=lora_alpha,
            target_modules=list(target_modules) if target_modules is not None else None,
            fan_in_fan_out=fan_in_fan_out,
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

    @property
    def is_causal_lm(self) -> bool:
        return self.task_type.endswith("CAUSAL_LM")

    @staticmethod
    def _slice_value(value, indices, batch_size):
        if torch.is_tensor(value):
            return value[indices] if value.ndim > 0 and value.size(0) == batch_size else value
        if hasattr(value, "batch_split"):
            splits = value.batch_split(batch_size, 1)
            selected = [splits[index] for index in indices.tolist()]
            from_batch_splits = getattr(type(value), "from_batch_splits", None)
            if from_batch_splits is not None:
                return from_batch_splits(selected)
        if hasattr(value, "batch_select_indices"):
            selected = copy.deepcopy(value)
            selected.batch_select_indices(indices.to(selected.key_cache[0].device))
            return selected
        if isinstance(value, tuple):
            return tuple(
                HyperLoRAModel._slice_value(item, indices, batch_size)
                for item in value
            )
        return value

    @classmethod
    def _slice_batch(cls, kwargs, indices, batch_size):
        return {
            key: value
            if key in {"cache_position", "logits_to_keep"}
            else cls._slice_value(value, indices, batch_size)
            for key, value in kwargs.items()
        }

    @staticmethod
    def _slice_args(args, indices, batch_size):
        return tuple(
            value[indices]
            if torch.is_tensor(value) and value.ndim > 0 and value.size(0) == batch_size
            else value
            for value in args
        )

    def _causal_lm_positional_names(self):
        """Return positional parameters from the actual Transformers backbone."""
        backbone = self.model
        get_base_model = getattr(backbone, "get_base_model", None)
        if callable(get_base_model):
            backbone = get_base_model()
        else:
            backbone = getattr(getattr(backbone, "base_model", None), "model", backbone)

        try:
            signature = inspect.signature(backbone.forward)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Cannot inspect the causal-LM backbone signature; pass inputs by keyword"
            ) from exc

        parameters = tuple(signature.parameters.values())
        if any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL
            for parameter in parameters
        ):
            raise TypeError(
                "The causal-LM backbone has a variadic forward signature; "
                "pass inputs by keyword"
            )

        return tuple(
            parameter.name
            for parameter in parameters
            if parameter.name != "self"
            and parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )

    @staticmethod
    def _split_output_value(value, row, batch_size):
        if torch.is_tensor(value):
            return value[row : row + 1] if value.ndim > 0 and value.size(0) == batch_size else value
        if hasattr(value, "batch_split"):
            return value.batch_split(batch_size, 1)[row]
        if isinstance(value, tuple):
            return tuple(
                HyperLoRAModel._split_output_value(item, row, batch_size)
                for item in value
            )
        return value

    @staticmethod
    def _zero_batch_like(value, batch_size):
        if torch.is_tensor(value):
            return value.new_zeros((batch_size, *value.shape[1:]))
        if isinstance(value, tuple):
            return tuple(
                HyperLoRAModel._zero_batch_like(item, batch_size)
                for item in value
            )
        return value

    @classmethod
    def _pad_generation_output(
        cls, output, max_length, max_steps, pad_token_id, templates
    ):
        fields = {
            field_name: getattr(output, field_name)
            for field_name in output.__dataclass_fields__
        }
        sequences = fields["sequences"]
        sequence_length = sequences.size(1)
        if sequences.size(1) < max_length:
            padded = sequences.new_full(
                (sequences.size(0), max_length), pad_token_id
            )
            padded[:, : sequences.size(1)] = sequences
            fields["sequences"] = padded

        for field_name, value in fields.items():
            if (
                field_name != "sequences"
                and torch.is_tensor(value)
                and value.ndim > 1
                and value.size(0) == sequences.size(0)
                and value.size(1) == sequence_length
                and sequence_length < max_length
            ):
                fill_value = -1 if field_name == "beam_indices" else 0
                padded = value.new_full(
                    (value.size(0), max_length, *value.shape[2:]), fill_value
                )
                padded[:, :sequence_length] = value
                fields[field_name] = padded

        # Generation can stop at different steps for different annotator groups.
        # Keep optional per-step fields rectangular so they can be re-stacked.
        for field_name in ("scores", "logits", "attentions", "hidden_states"):
            values = fields.get(field_name)
            target_steps = max_steps.get(field_name, 0)
            if not isinstance(values, tuple) or len(values) >= target_steps:
                continue
            template = next(
                (
                    getattr(other, field_name)[-1]
                    for other in templates
                    if isinstance(getattr(other, field_name, None), tuple)
                    and getattr(other, field_name)
                ),
                None,
            )
            if template is not None:
                values = values + tuple(
                    cls._zero_batch_like(template, sequences.size(0))
                    for _ in range(target_steps - len(values))
                )
                fields[field_name] = values

        return type(output)(**fields)

    def _stack_causal_outputs(self, grouped_outputs, batch_size):
        outputs_by_row = [None] * batch_size
        for batch_indices, output in grouped_outputs:
            for row, batch_index in enumerate(batch_indices.tolist()):
                fields = {
                    field_name: self._split_output_value(
                        getattr(output, field_name), row, batch_indices.size(0)
                    )
                    for field_name in output.__dataclass_fields__
                }
                outputs_by_row[batch_index] = type(output)(**fields)

        from transformers.generation.utils import stack_model_outputs

        return stack_model_outputs(outputs_by_row, self.base_model.config)

    def forward(self, *args, **kwargs):
        # pop off the hypernetwork IDs
        HN_ids = kwargs.pop("annotator_ids").to(self.device)
        batch = HN_ids.size(0)
        if self.is_causal_lm and args:
            positional_names = self._causal_lm_positional_names()
            if len(args) > len(positional_names):
                raise TypeError("Too many positional arguments for a causal LM")
            for name, value in zip(positional_names, args):
                if name in kwargs:
                    raise TypeError(f"{name} was passed both positionally and by keyword")
                kwargs[name] = value
            args = ()
        labels = kwargs.pop("labels", None)
        kwargs = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in kwargs.items()
        }
        requested_return_dict = None
        if self.is_causal_lm:
            requested_return_dict = kwargs.get(
                "return_dict", getattr(self.base_model.config, "return_dict", True)
            )
            kwargs["return_dict"] = True

        # Group batch items by annotator ID to avoid redundant LoRA weight generation
        unique_ids, inverse_indices = torch.unique(HN_ids, return_inverse=True)

        # Generate LoRA weights for unique annotator IDs only
        weights_unique = self.hypernet(unique_ids)

        logits = None
        grouped_outputs = []

        for uid_idx in range(unique_ids.size(0)):
            # Find all batch indices sharing this annotator ID
            mask = inverse_indices == uid_idx
            batch_indices = mask.nonzero(as_tuple=True)[0]

            group_weights = [
                (A_group[uid_idx], B_group[uid_idx])
                for A_group, B_group in weights_unique
            ]

            with self._inject_lora_weights(group_weights):
                sub_kwargs = self._slice_batch(kwargs, batch_indices, batch)
                sub_args = self._slice_args(args, batch_indices, batch)
                out = self.base_model(*sub_args, **sub_kwargs)
                if self.is_causal_lm:
                    grouped_outputs.append((batch_indices, out))
                out_logits = out.logits if hasattr(out, "logits") else out[0]
                if logits is None:
                    logits = out_logits.new_empty((batch, *out_logits.shape[1:]))
                logits[batch_indices] = out_logits

        if logits is None:
            raise RuntimeError("Cannot run HyperLoRAModel with an empty batch")

        if self.is_causal_lm:
            result = self._stack_causal_outputs(grouped_outputs, batch)
        else:
            # Classification metrics use the prepended annotator ID.
            result = {"logits": torch.cat([HN_ids.unsqueeze(-1), logits], dim=-1)}
        if labels is not None:
            labels = labels.to(self.device)
            if self.is_causal_lm:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            else:
                loss = self.loss(logits, labels)
            if self.is_causal_lm:
                fields = {
                    field_name: getattr(result, field_name)
                    for field_name in result.__dataclass_fields__
                }
                fields["loss"] = loss
                result = type(result)(**fields)
            else:
                result["loss"] = loss

        if self.is_causal_lm and requested_return_dict is False:
            return result.to_tuple()
        return result

    def generate(self, *args, **kwargs):
        """Generate with annotator-specific LoRA weights held for each sequence."""
        if not self.is_causal_lm:
            raise TypeError("generate() is only available when task_type='CAUSAL_LM'")
        if len(args) > 1 or (args and "input_ids" in kwargs):
            raise TypeError("Pass input_ids either positionally or by keyword")
        if args:
            kwargs["input_ids"] = args[0]

        generation_config = kwargs.get("generation_config")
        if generation_config is None:
            generation_config = getattr(self.base_model, "generation_config", None)
        num_return_sequences = kwargs.get("num_return_sequences")
        if num_return_sequences is None:
            num_return_sequences = getattr(generation_config, "num_return_sequences", 1)
        return_dict_in_generate = kwargs.get("return_dict_in_generate")
        if return_dict_in_generate is None:
            return_dict_in_generate = getattr(
                generation_config, "return_dict_in_generate", False
            )

        HN_ids = kwargs.pop("annotator_ids").to(self.device)
        kwargs = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in kwargs.items()
        }
        batch = HN_ids.size(0)
        unique_ids, inverse_indices = torch.unique(HN_ids, return_inverse=True)
        weights_unique = self.hypernet(unique_ids)
        generated = []
        group_targets = []

        for uid_idx in range(unique_ids.size(0)):
            indices = (inverse_indices == uid_idx).nonzero(as_tuple=True)[0]
            sub_kwargs = self._slice_batch(kwargs, indices, batch)
            group_weights = [
                (A_group[uid_idx], B_group[uid_idx])
                for A_group, B_group in weights_unique
            ]
            with self._inject_lora_weights(group_weights):
                output = self.base_model.generate(**sub_kwargs)
            sequences = output.sequences if return_dict_in_generate else output
            if not torch.is_tensor(sequences):
                raise TypeError("Causal-LM generation must return token sequences")
            if sequences.size(0) % indices.size(0) != 0:
                raise ValueError("Generation output batch does not match input batch")
            group_return_sequences = sequences.size(0) // indices.size(0)
            if group_return_sequences != num_return_sequences:
                raise ValueError(
                    "num_return_sequences does not match the generation output"
                )
            for row, batch_index in enumerate(indices.tolist()):
                for candidate in range(num_return_sequences):
                    group_targets.append(batch_index * num_return_sequences + candidate)
            generated.append(output)

        max_length = max(
            (output.sequences if return_dict_in_generate else output).size(1)
            for output in generated
        )
        pad_token_id = self.base_model.config.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.base_model.config.eos_token_id or 0
            if isinstance(pad_token_id, (list, tuple)):
                pad_token_id = pad_token_id[0]

        if not return_dict_in_generate:
            current_sequences = []
            for output in generated:
                if output.size(1) < max_length:
                    padded = output.new_full(
                        (output.size(0), max_length), pad_token_id
                    )
                    padded[:, : output.size(1)] = output
                    output = padded
                current_sequences.append(output)
            all_sequences = torch.cat(current_sequences, dim=0)
            target_to_current = {
                target: row for row, target in enumerate(group_targets)
            }
            order = [target_to_current[target] for target in range(batch * num_return_sequences)]
            return all_sequences[torch.tensor(order, device=all_sequences.device)]

        from transformers.generation.utils import stack_model_outputs

        max_steps = {
            field_name: max(
                len(getattr(output, field_name, ()) or ()) for output in generated
            )
            for field_name in ("scores", "logits", "attentions", "hidden_states")
        }
        normalized_outputs = [
            self._pad_generation_output(
                output,
                max_length,
                max_steps,
                pad_token_id,
                generated,
            )
            for output in generated
        ]

        result = stack_model_outputs(normalized_outputs, self.base_model.config)
        target_to_current = {
            target: row for row, target in enumerate(group_targets)
        }
        order = torch.tensor(
            [target_to_current[target] for target in range(batch * num_return_sequences)],
            device=result.sequences.device,
        )

        def reorder(value):
            if torch.is_tensor(value):
                return (
                    value.index_select(0, order)
                    if value.ndim > 0 and value.size(0) == len(group_targets)
                    else value
                )
            if hasattr(value, "batch_select_indices"):
                value.batch_select_indices(order)
                return value
            if isinstance(value, tuple):
                return tuple(reorder(item) for item in value)
            return value

        fields = {
            field_name: reorder(getattr(result, field_name))
            for field_name in result.__dataclass_fields__
        }
        return type(result)(**fields)
