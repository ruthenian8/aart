"""Tests for tokenization (batched), model forward, and majority inference."""
import sys
import pytest
import torch
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _hf_hub_available():
    """Check if HuggingFace Hub is reachable."""
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("roberta-base")
        return True
    except Exception:
        return False


requires_hf_hub = pytest.mark.skipif(
    not _hf_hub_available(),
    reason="HuggingFace Hub is not reachable in this environment"
)


@requires_hf_hub
class TestBatchedTokenization:
    """Test that batched tokenization produces correct output."""

    @pytest.fixture
    def tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("roberta-base")

    def test_single_text_tokenization(self, tokenizer):
        """Batched tokenization for single-text datasets."""
        batch = {"text": ["Hello world", "Foo bar baz"]}
        result = tokenizer(
            text=batch["text"],
            padding="max_length",
            truncation=True,
            max_length=32,
        )
        assert "input_ids" in result
        assert len(result["input_ids"]) == 2
        assert len(result["input_ids"][0]) == 32

    def test_pair_text_tokenization(self, tokenizer):
        """Batched tokenization for pair-text datasets."""
        batch = {
            "parent_text": ["Parent text one", "Parent text two"],
            "text": ["Child text one", "Child text two"],
        }
        result = tokenizer(
            text=batch["parent_text"],
            text_pair=batch["text"],
            padding="max_length",
            truncation=True,
            max_length=64,
        )
        assert "input_ids" in result
        assert len(result["input_ids"]) == 2
        assert len(result["input_ids"][0]) == 64


@requires_hf_hub
class TestModelForward:
    """Test HyperLoRAModel forward pass."""

    @pytest.fixture
    def model(self):
        from model_architectures import HyperLoRAModel
        m = HyperLoRAModel.from_pretrained(
            "roberta-base",
            num_labels=2,
            num_embeddings=10,
            lora_r=2,
            device=torch.device("cpu"),
        )
        m.eval()
        return m

    @pytest.fixture
    def dummy_inputs(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("roberta-base")
        encoded = tok(
            ["Hello world", "Foo bar"],
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "annotator_ids": torch.tensor([0, 1]),
        }

    def test_forward_with_labels(self, model, dummy_inputs):
        dummy_inputs["labels"] = torch.tensor([0, 1])
        with torch.no_grad():
            result = model(**dummy_inputs)
        assert "loss" in result
        assert "logits" in result
        # logits should be (batch, 1 + num_labels) because annotator ID is prepended
        assert result["logits"].shape == (2, 3)  # 1 annotator_id + 2 class logits

    def test_forward_without_labels(self, model, dummy_inputs):
        """Forward should work without labels (inference mode)."""
        with torch.no_grad():
            result = model(**dummy_inputs)
        assert "loss" not in result
        assert "logits" in result
        assert result["logits"].shape == (2, 3)

    def test_forward_grouped_by_annotator(self, model):
        """When batch items share an annotator ID, they should be grouped."""
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("roberta-base")
        encoded = tok(
            ["text one", "text two", "text three"],
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )
        inputs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            # Two items share annotator 0
            "annotator_ids": torch.tensor([0, 0, 1]),
            "labels": torch.tensor([0, 1, 0]),
        }
        with torch.no_grad():
            result = model(**inputs)
        assert result["logits"].shape == (3, 3)
        assert "loss" in result


class TestUtils:
    """Test utility functions."""

    def test_extract_model_logits_ndarray(self):
        from utils import extract_model_logits
        from types import SimpleNamespace

        preds = SimpleNamespace(predictions=np.array([[1.0, 2.0], [3.0, 4.0]]))
        result = extract_model_logits(preds)
        assert result.shape == (2, 2)

    def test_extract_model_logits_tuple(self):
        from utils import extract_model_logits
        from types import SimpleNamespace

        raw = (np.array([[1.0, 2.0]]), np.array([[5.0]]))
        preds = SimpleNamespace(predictions=raw)
        result = extract_model_logits(preds)
        assert result.shape == (1, 2)

    def test_macro_f1_fallback_empty(self):
        """When no annotator has > 5 examples, macro_f1 should not be NaN."""
        from utils import _compute_metrics_hpm

        # 4 examples with 1 annotator (below threshold of 5)
        logits = np.array([
            [0, 2.0, 0.1],
            [0, 0.1, 2.0],
            [0, 2.0, 0.1],
            [0, 0.1, 2.0],
        ])
        labels = np.array([0, 1, 0, 1])
        result = _compute_metrics_hpm((logits, labels))
        assert not np.isnan(result["macro_f1"])

    def test_get_a_p_r_f_basic(self):
        from utils import get_a_p_r_f
        a, p, r, f = get_a_p_r_f([0, 1, 0, 1], [0, 1, 0, 1])
        assert a == 100.0
        assert f == 100.0
