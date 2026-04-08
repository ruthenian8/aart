import logging
import math
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)


@dataclass
class Params:
    """Validated configuration object for the HPM pipeline."""

    num_classes: int = 2
    data_name: Optional[str] = None
    language_model_name: str = "roberta-base"
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_len: int = 128
    num_epochs: int = 20
    random_state: int = 2023
    approach: str = "hpm"
    skip_test: bool = False
    use_majority_weight: bool = True
    majority_inference: bool = False
    lambda2: float = 0.0
    balance_annotator_weights: bool = False
    early_stopping_patience: int = 15
    embedding_colnames: str = ""
    sort_instances_by: str = ""
    num_fake_annotators: int = 0
    contrastive_alpha: float = 0.0
    device: Optional[str] = None
    num_tokenization_workers: Optional[int] = None
    deterministic: bool = True
    unknown_annotator_policy: str = "drop"

    @classmethod
    def from_namespace(cls, args) -> "Params":
        """Create a Params instance from an argparse Namespace, validating values."""
        p = cls()
        for k, v in vars(args).items():
            if hasattr(p, k) and v is not None:
                # Skip NaN float values from argparse defaults
                if isinstance(v, float) and math.isnan(v):
                    continue
                logger.debug("Setting %s = %s (was %s)", k, v, getattr(p, k))
                setattr(p, k, v)

        p.validate()
        return p

    def validate(self):
        """Validate parameter values, raising ValueError for invalid configs."""
        if self.approach != "hpm":
            raise ValueError(
                f"Unsupported approach '{self.approach}'. Only 'hpm' is currently supported."
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.max_len < 1:
            raise ValueError(f"max_len must be positive, got {self.max_len}")
        if self.num_epochs < 0:
            raise ValueError(f"num_epochs must be non-negative, got {self.num_epochs}")
        if self.num_tokenization_workers is not None and self.num_tokenization_workers < 0:
            raise ValueError(
                f"num_tokenization_workers must be non-negative, got {self.num_tokenization_workers}"
            )
        if self.unknown_annotator_policy not in ("error", "drop"):
            raise ValueError(
                f"unknown_annotator_policy must be 'error' or 'drop', got '{self.unknown_annotator_policy}'"
            )

    def __str__(self):
        return str(self.__dict__)
