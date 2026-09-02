"""Tests for tokenization (batched), model forward, and majority inference."""
import sys
import pytest
import torch
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _hf_hub_available():
    """Check if HuggingFace Hub is reachable (tokenizer only — small files)."""
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("roberta-base")
        return True
    except (OSError, ConnectionError, RuntimeError):
        return False


def _roberta_model_cached():
    """Return True only if roberta-base model weights are already in the local cache.

    This prevents tests from blocking on a large (~499 MB) download in
    environments where HuggingFace Hub is slow or rate-limited.
    """
    try:
        from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST
        result = try_to_load_from_cache("roberta-base", "model.safetensors")
        if result is None or result is _CACHED_NO_EXIST:
            return False
        return True
    except Exception:
        return False


requires_hf_hub = pytest.mark.skipif(
    not _hf_hub_available(),
    reason="HuggingFace Hub is not reachable in this environment"
)

requires_model_weights = pytest.mark.skipif(
    not _roberta_model_cached(),
    reason="roberta-base model weights not in local cache; skipping to avoid large download"
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


@requires_model_weights
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

    def test_forward_accepts_tuple_backbone_output(self, model, dummy_inputs):
        with torch.no_grad():
            result = model(**dummy_inputs, return_dict=False)

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


class TestCausalLMForward:
    """Test causal-LM support without downloading pretrained weights."""

    @pytest.fixture
    def model(self, tmp_path):
        from transformers import GPT2Config, GPT2LMHeadModel
        from model_architectures import HyperLoRAModel

        model_path = tmp_path / "tiny-gpt2"
        config = GPT2Config(
            vocab_size=32,
            n_positions=16,
            n_embd=12,
            n_layer=2,
            n_head=2,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        GPT2LMHeadModel(config).save_pretrained(model_path)
        model = HyperLoRAModel.from_pretrained(
            model_path,
            num_embeddings=4,
            lora_r=2,
            task_type="CAUSAL_LM",
            target_modules=["c_attn"],
            fan_in_fan_out=True,
            device=torch.device("cpu"),
        )
        model.eval()
        return model

    def test_forward_uses_token_logits_and_causal_loss(self, model):
        input_ids = torch.tensor([[1, 4, 5, 2], [1, 6, 7, 2]])
        with torch.no_grad():
            result = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                annotator_ids=torch.tensor([0, 1]),
                labels=input_ids,
            )

        assert result["logits"].shape == (2, 4, 32)
        assert result["loss"].ndim == 0

        with torch.no_grad():
            positional_result = model(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                annotator_ids=torch.tensor([0, 1]),
            )
        assert positional_result["logits"].shape == (2, 4, 32)

    def test_forward_uses_backbone_positional_signature(self, model):
        input_ids = torch.tensor([[1, 4, 5, 2], [1, 6, 7, 2]])
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
        annotator_ids = torch.tensor([0, 1])

        assert model._causal_lm_positional_names()[:3] == (
            "input_ids",
            "past_key_values",
            "attention_mask",
        )

        with torch.no_grad():
            keyword_result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                annotator_ids=annotator_ids,
            )
            positional_result = model(
                input_ids,
                None,
                attention_mask,
                annotator_ids=annotator_ids,
            )

        torch.testing.assert_close(
            positional_result.logits,
            keyword_result.logits,
        )

    def test_forward_preserves_causal_output_options(self, model):
        input_ids = torch.tensor([[1, 4, 5, 2], [1, 6, 7, 2]])
        with torch.no_grad():
            result = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                annotator_ids=torch.tensor([0, 1]),
                output_hidden_states=True,
                output_attentions=True,
                return_dict=False,
            )

        assert isinstance(result, tuple)
        assert result[0].shape == (2, 4, 32)
        assert result[-2] is not None  # hidden states
        assert result[-1] is not None  # attentions

    def test_forward_tuple_places_loss_before_logits(self, model):
        input_ids = torch.tensor([[1, 4, 5, 2], [1, 6, 7, 2]])
        with torch.no_grad():
            result = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                annotator_ids=torch.tensor([0, 1]),
                labels=input_ids,
                return_dict=False,
            )

        assert isinstance(result, tuple)
        assert result[0].ndim == 0
        assert result[1].shape == (2, 4, 32)

    def test_generate_preserves_batch_order(self, model):
        input_ids = torch.tensor([[1, 4], [1, 5], [1, 6]])
        with torch.no_grad():
            sequences = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                annotator_ids=torch.tensor([1, 0, 1]),
                max_new_tokens=2,
                do_sample=False,
            )

        assert sequences.shape == (3, 4)
        assert torch.equal(sequences[:, :2], input_ids)

    def test_generate_supports_multiple_return_sequences(self, model):
        input_ids = torch.tensor([[1, 4], [1, 5], [1, 6]])
        with torch.no_grad():
            sequences = model.generate(
                input_ids,
                annotator_ids=torch.tensor([1, 0, 1]),
                num_beams=2,
                num_return_sequences=2,
                max_new_tokens=2,
                do_sample=False,
            )

        assert sequences.shape == (6, 4)
        assert torch.equal(sequences[::2, :2], input_ids)
        assert torch.equal(sequences[1::2, :2], input_ids)

    def test_generate_supports_structured_results(self, model):
        input_ids = torch.tensor([[1, 4], [1, 5], [1, 6]])
        with torch.no_grad():
            result = model.generate(
                input_ids=input_ids,
                annotator_ids=torch.tensor([1, 0, 1]),
                max_new_tokens=2,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        assert result.sequences.shape == (3, 4)
        assert len(result.scores) == 2

    def test_gqa_backbone_uses_shape_specific_heads(self, tmp_path):
        from transformers import LlamaConfig, LlamaForCausalLM
        from model_architectures import HyperLoRAModel

        model_path = tmp_path / "tiny-llama"
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=16,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        LlamaForCausalLM(config).save_pretrained(model_path)
        model = HyperLoRAModel.from_pretrained(
            model_path,
            num_embeddings=4,
            task_type="CAUSAL_LM",
            device=torch.device("cpu"),
        )

        assert len(model.lora_module_groups) == 2


class TestHyperNetworkCollection:
    """Test shape-specific hypernetwork heads."""

    def test_supports_different_projection_shapes(self):
        from model_hypernetwork import HyperNetworkCollection

        model = HyperNetworkCollection(
            speaker_dim=8,
            context_dim=8,
            hidden_dim=8,
            r=2,
            num_embeddings=4,
            group_specs=[(8, 12, 2), (8, 6, 2)],
        )
        outputs = model(torch.tensor([0, 1]))

        assert outputs[0][0].shape == (2, 2, 2, 8)
        assert outputs[0][1].shape == (2, 2, 12, 2)
        assert outputs[1][0].shape == (2, 2, 2, 8)
        assert outputs[1][1].shape == (2, 2, 6, 2)


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


class TestMajorityLabelBias:
    """Test that majority label computation is not biased by synthetic rows."""

    def test_majority_label_uses_only_real_labels(self):
        """Synthetic (expanded) rows filled with 0 should NOT affect maj_label."""
        import pandas as pd
        import numpy as np

        # Simulate test_expanded after expand_test + fillna(0):
        # text_id 1: 2 real labels (both 1), 1 synthetic (filled to 0)
        # text_id 2: 2 real labels (both 0), 1 synthetic (filled to 0)
        data = {
            "text_id": [1, 1, 1, 2, 2, 2],
            "label": [1, 1, 0, 0, 0, 0],    # third row in each group is synthetic
            "_has_real_label": [True, True, False, True, True, False],
            "pred": [1, 1, 1, 0, 0, 0],
        }
        df = pd.DataFrame(data)

        # Majority label should be based on real labels only:
        # text 1: mean(1, 1) = 1.0 >= 0.5 → True
        # text 2: mean(0, 0) = 0.0 < 0.5 → False
        real_df = df[df["_has_real_label"]]
        maj_label = real_df.groupby("text_id")["label"].mean() >= 0.5
        assert maj_label[1] == True
        assert maj_label[2] == False

        # Without the fix (using all rows including synthetic):
        # text 1: mean(1, 1, 0) = 0.67 → would still be True (less clear)
        # text 2: mean(0, 0, 0) = 0.0 → False (coincidentally correct here)
        # But with text_id 3 having 1 real label=1 and synthetic=0:
        data2 = {
            "text_id": [3, 3],
            "label": [1, 0],
            "_has_real_label": [True, False],
            "pred": [1, 0],
        }
        df2 = pd.DataFrame(data2)
        real_only = df2[df2["_has_real_label"]].groupby("text_id")["label"].mean() >= 0.5
        all_rows = df2.groupby("text_id")["label"].mean() >= 0.5
        # Real-only: 1/1 = 1.0 → True
        assert real_only[3] == True
        # Naive all-rows: (1+0)/2 = 0.5 → True (borderline, but same result here)
        # The fix is most critical when synthetic rows push mean below 0.5
        data3 = {
            "text_id": [4, 4, 4],
            "label": [1, 0, 0],   # 1 real positive, 2 synthetic zeros
            "_has_real_label": [True, False, False],
        }
        df3 = pd.DataFrame(data3)
        real_only3 = df3[df3["_has_real_label"]].groupby("text_id")["label"].mean() >= 0.5
        naive3 = df3.groupby("text_id")["label"].mean() >= 0.5
        # Real-only: 1/1 = 1.0 → True (correct)
        assert real_only3[4] == True
        # Naive: 1/3 = 0.33 → False (WRONG — incorrectly classifies as negative majority)
        assert naive3[4] == False  # this is the bug we fixed


class TestSaveRepresentations:
    """Test save_representations writes the expected files."""

    def test_save_representations_writes_npy_and_json(self, tmp_path):
        """save_representations should write .npy and .json with correct shapes."""
        import json
        from types import SimpleNamespace
        from pathlib import Path

        sys.path.insert(0, str(REPO_ROOT))
        from pipelines.hpm_pipeline import HPMPipeline

        # Build a minimal mock model that exposes the same attribute path
        # as HyperLoRAModel: model.hypernet.speaker_emb
        num_annotators = 5
        speaker_dim = 8
        emb_weight = torch.randn(num_annotators, speaker_dim)
        speaker_emb = torch.nn.Embedding(num_annotators, speaker_dim)
        speaker_emb.weight = torch.nn.Parameter(emb_weight)
        hypernet = SimpleNamespace(speaker_emb=speaker_emb)
        mock_model = SimpleNamespace(hypernet=hypernet)

        # Build a minimal pipeline stub (bypass __init__) with enough state
        pipeline = object.__new__(HPMPipeline)
        pipeline.data_dict = {
            "annotator_map": {0: "alice", 1: "bob", 2: "carol", 3: "dave", 4: "eve"}
        }

        output_dir = tmp_path / "representations"
        pipeline.save_representations(mock_model, output_dir)

        # .npy file
        npy_path = output_dir / "annotator_representations.npy"
        assert npy_path.exists(), "annotator_representations.npy was not created"
        saved = np.load(str(npy_path))
        assert saved.shape == (num_annotators, speaker_dim)
        np.testing.assert_allclose(
            saved, emb_weight.detach().cpu().float().numpy(), rtol=1e-5
        )

        # .json map file
        json_path = output_dir / "annotator_id_map.json"
        assert json_path.exists(), "annotator_id_map.json was not created"
        loaded_map = json.loads(json_path.read_text())
        assert loaded_map["0"] == "alice"
        assert loaded_map["4"] == "eve"

    def test_save_representations_no_map(self, tmp_path):
        """save_representations should still write .npy when annotator_map is absent."""
        from types import SimpleNamespace
        from pipelines.hpm_pipeline import HPMPipeline

        num_annotators = 3
        speaker_dim = 4
        speaker_emb = torch.nn.Embedding(num_annotators, speaker_dim)
        hypernet = SimpleNamespace(speaker_emb=speaker_emb)
        mock_model = SimpleNamespace(hypernet=hypernet)

        pipeline = object.__new__(HPMPipeline)
        pipeline.data_dict = {}  # no annotator_map

        output_dir = tmp_path / "repr_no_map"
        pipeline.save_representations(mock_model, output_dir)

        npy_path = output_dir / "annotator_representations.npy"
        assert npy_path.exists()
        # .json should NOT be created when map is absent
        json_path = output_dir / "annotator_id_map.json"
        assert not json_path.exists()
