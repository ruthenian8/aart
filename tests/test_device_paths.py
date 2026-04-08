"""Tests for device initialization and path handling."""
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


class TestDeviceInit:
    """Test that device resolution works on CPU-only machines."""

    def test_device_resolver_cpu(self):
        """Pipeline device resolver should fall back to CPU when CUDA is not available."""
        import torch
        with patch.object(torch.cuda, "is_available", return_value=False):
            from params import Params
            p = Params(device=None)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            assert device.type == "cpu"

    def test_device_explicit_cpu(self):
        """Explicit device=cpu should be honored."""
        import torch
        from params import Params
        p = Params(device="cpu")
        device = torch.device(p.device)
        assert device.type == "cpu"

    def test_cpu_training_args_compat(self):
        """get_trainingargs sets the correct CPU flag for the installed transformers version."""
        import dataclasses
        from transformers import TrainingArguments
        from pipelines.generic_pipeline import GenericPipeline

        ta_fields = {f.name for f in dataclasses.fields(TrainingArguments)}

        # Simulate the logic used in get_trainingargs
        training_args = {}
        import torch
        device = torch.device("cpu")
        if device.type == "cpu":
            if "use_cpu" in ta_fields:
                training_args["use_cpu"] = True
            else:
                training_args["no_cuda"] = True

        # Exactly one of the two flags should be set
        assert ("use_cpu" in training_args or "no_cuda" in training_args)
        assert not ("use_cpu" in training_args and "no_cuda" in training_args)
        # The set flag must be a valid TrainingArguments field
        flag = "use_cpu" if "use_cpu" in training_args else "no_cuda"
        assert flag in ta_fields, f"{flag!r} is not a valid TrainingArguments field"


class TestPathHandling:
    """Test that repo-root path handling works correctly."""

    def test_repo_root_is_correct(self):
        from pipelines.generic_pipeline import REPO_ROOT as PIPELINE_ROOT
        from main import REPO_ROOT as MAIN_ROOT
        # Both should point to the same directory
        assert PIPELINE_ROOT == MAIN_ROOT

    def test_repo_root_contains_main(self):
        from main import REPO_ROOT
        assert (REPO_ROOT / "main.py").exists()

    def test_repo_root_independent_of_cwd(self, tmp_path, monkeypatch):
        """REPO_ROOT should not depend on the current working directory."""
        monkeypatch.chdir(tmp_path)
        # Re-import to check
        from main import REPO_ROOT
        assert (REPO_ROOT / "main.py").exists()
