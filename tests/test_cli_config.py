"""Tests for CLI argument parsing and configuration."""
import sys
import pytest
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from params import Params


class TestParams:
    def test_default_params(self):
        p = Params()
        assert p.approach == "hpm"
        assert p.batch_size == 16
        assert p.max_len == 128

    def test_valid_hpm_approach(self):
        p = Params(approach="hpm")
        p.validate()  # should not raise

    def test_invalid_approach_raises(self):
        p = Params(approach="single")
        with pytest.raises(ValueError, match="Unsupported approach"):
            p.validate()

    def test_invalid_batch_size_raises(self):
        p = Params(batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            p.validate()

    def test_invalid_max_len_raises(self):
        p = Params(max_len=-1)
        with pytest.raises(ValueError, match="max_len must be positive"):
            p.validate()

    def test_invalid_num_epochs_raises(self):
        p = Params(num_epochs=-1)
        with pytest.raises(ValueError, match="num_epochs must be non-negative"):
            p.validate()

    def test_invalid_unknown_annotator_policy_raises(self):
        p = Params(unknown_annotator_policy="ignore")
        with pytest.raises(ValueError, match="unknown_annotator_policy"):
            p.validate()

    def test_from_namespace(self):
        import argparse
        ns = argparse.Namespace(
            data_name="test_data",
            language_model_name="roberta-base",
            approach="hpm",
            max_len=64,
            batch_size=8,
            learning_rate=1e-5,
            num_epochs=5,
            random_state=42,
            skip_test=False,
            majority_inference=False,
            lambda2=0.0,
            contrastive_alpha=0.0,
            embedding_colnames="",
            sort_instances_by="",
            num_fake_annotators=0,
        )
        p = Params.from_namespace(ns)
        assert p.data_name == "test_data"
        assert p.max_len == 64
        assert p.batch_size == 8


class TestCLI:
    def test_parse_args_accepts_hpm(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["main.py", "--data_name", "test", "--approach", "hpm",
             "--language_model_name", "roberta-base", "--max_len", "64",
             "--batch_size", "8", "--learning_rate", "1e-5", "--num_epochs", "1"]
        )
        from main import parse_args
        args = parse_args()
        assert args.approach == "hpm"

    def test_parse_args_rejects_invalid(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["main.py", "--data_name", "test", "--approach", "single",
             "--language_model_name", "roberta-base", "--max_len", "64",
             "--batch_size", "8", "--learning_rate", "1e-5", "--num_epochs", "1"]
        )
        from main import parse_args
        with pytest.raises(SystemExit):
            parse_args()

    def test_get_pipeline_unsupported_raises(self):
        from main import get_pipeline
        p = Params(approach="multi_task")
        # Skip validation to test get_pipeline directly
        p.approach = "multi_task"
        with pytest.raises(ValueError, match="Unsupported approach"):
            get_pipeline(p)


class TestSlugify:
    def test_basic_slugify(self):
        from main import slugify_experiment_name
        result = slugify_experiment_name("hello/world: foo, bar")
        assert "/" not in result
        assert ":" not in result
        assert "," not in result
        assert " " not in result

    def test_max_length(self):
        from main import slugify_experiment_name
        result = slugify_experiment_name("a" * 300, max_length=50)
        assert len(result) <= 50
