"""Explicit single-GPU analytic witness for the native compiled entropy."""
from pathlib import Path
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot_utils import ROOT
sys.path.insert(0, str(ROOT / ".runtime/upstream/Agent0/Agent0-VL"))
torch = pytest.importorskip("torch")


@pytest.mark.skipif(os.environ.get("BASELINE_GPU_DIAGNOSTIC") != "1", reason="Explicit selected GPU diagnostic required")
def test_compiled_native_entropy_matches_analytic_values_and_derivatives():
    from verl.utils.torch_functional import entropy_from_logits
    values = np.linspace(-2, 2, 5 * 1024, dtype=np.float64).reshape(5, 1024)
    probabilities = np.exp(values - values.max(-1, keepdims=True))
    probabilities /= probabilities.sum(-1, keepdims=True)
    expected = -(probabilities * np.log(probabilities)).sum(-1)
    derivatives = -probabilities * (values - (probabilities * values).sum(-1, keepdims=True))
    actual = torch.tensor(values, dtype=torch.float32, device="cuda", requires_grad=True)
    result = torch.compile(entropy_from_logits, dynamic=True)(actual)
    result.sum().backward()
    np.testing.assert_allclose(result.detach().cpu().numpy(), expected, rtol=5e-6, atol=1e-6)
    np.testing.assert_allclose(actual.grad.cpu().numpy(), derivatives, rtol=2e-4, atol=1e-7)
