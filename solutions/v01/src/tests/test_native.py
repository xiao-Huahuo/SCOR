"""Run against the real pinned native reward, GRPO and tool sandbox on the server."""
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot_utils import ROOT
sys.path.insert(0, str(ROOT / ".runtime/upstream/Agent0/Agent0-VL"))

import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")
pytest.importorskip("math_verify")
from tensordict import TensorDict
import numpy as np
from verl import DataProto
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
from verl.workers.reward_manager.agent0 import Agent0RewardManager
from sandbox.internal_sandbox import parallel_sandbox
from scor_hooks import apply_corruption, sample_parameters, torch_save_immutable


def test_native_reward_and_entire_group_advantage_recompute():
    class Tokenizer:
        def decode(self, tokens, **kwargs):
            return "\\boxed{5}"

    steps = np.empty(4, dtype=object)
    for i in range(4):
        steps[i] = [{"score": 0.8, "confidence": 0.9, "tool_success": True,
                     "was_repaired": False, "verified": True, "step_end_pos": 2}]
    data = DataProto(batch=TensorDict({"responses": torch.zeros(4, 4, dtype=torch.long),
                                      "attention_mask": torch.ones(4, 8, dtype=torch.long)}, batch_size=[4]),
                     non_tensor_batch={"step_data": steps,
                                       "reward_model": np.array([{"ground_truth": "5"}] * 4, dtype=object),
                                       "final_answers": np.array(["5", "8", "5", "8"], dtype=object)})
    manager = Agent0RewardManager(Tokenizer(), num_examine=0,
                                  config={"lambda_tool": 0.3, "alpha_out": 1.0})
    rewards = manager(data)
    expected = torch.tensor([2.02, 1.02, 2.02, 1.02])
    assert torch.allclose(rewards.sum(-1), expected, atol=1e-5)
    corrupted = apply_corruption(rewards, data.batch["attention_mask"], 0, -3.0)
    assert torch.allclose(rewards.sum(-1), expected, atol=1e-5)
    assert torch.allclose(corrupted.sum(-1), torch.tensor([-0.98, 1.02, 2.02, 1.02]), atol=1e-5)
    group = np.array(["same-prompt"] * 4, dtype=object)
    mask = torch.ones_like(rewards)
    clean_adv, _ = compute_grpo_outcome_advantage(rewards.clone(), mask, group)
    corrupt_adv, _ = compute_grpo_outcome_advantage(corrupted.clone(), mask, group)
    assert torch.isfinite(corrupt_adv).all()
    assert clean_adv[0, 0] > 0 and corrupt_adv[0, 0] < 0
    assert torch.all(clean_adv[:, 0] != corrupt_adv[:, 0])


def test_real_native_tool_success_and_failure():
    success, outputs, errors = asyncio.run(parallel_sandbox(
        ["print(6 * 7)", "raise ValueError('expected-test-failure')"], num_processes=2))
    assert success == [True, False]
    assert outputs[0].strip() == "42"
    assert "expected-test-failure" in errors[1]


def test_actual_torch_receipt_is_loadable_and_immutable(tmp_path):
    path = tmp_path / "receipt.pt"
    torch_save_immutable({"values": torch.tensor([1.0, 2.0])}, path)
    loaded = torch.load(path, weights_only=True)
    assert torch.equal(loaded["values"], torch.tensor([1.0, 2.0]))
    with pytest.raises(FileExistsError):
        torch_save_immutable({"values": torch.tensor([9.0])}, path)
    assert torch.equal(torch.load(path, weights_only=True)["values"], loaded["values"])
    assert not list(tmp_path.glob("pending-*.pt"))


def test_parameter_probe_handles_shards_larger_than_float32_integer_precision():
    module = torch.nn.Module()
    module.register_parameter("large", torch.nn.Parameter(torch.zeros(20_000_000)))
    with torch.no_grad():
        module.large[0] = 3
        module.large[-1] = 7
    result = sample_parameters(module)
    assert result.shape == (64,)
    assert result[0] == 3 and result[-1] == 7
    assert torch.isfinite(result).all()
    assert module.large[0] == 3 and module.large[-1] == 7
