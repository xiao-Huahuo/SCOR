"""CPU native-function witnesses; actual four-GPU acceptance is separate."""
from pathlib import Path
import random
import statistics
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot_utils import ROOT, SRC, atomic_json, file_digest
sys.path.insert(0, str(ROOT / ".runtime/upstream/Agent0/Agent0-VL"))
torch = pytest.importorskip("torch")
from tensordict import TensorDict
from verl import DataProto
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
from interventions import false_positive
from native_adapter import rng_state, restore_rng, restore_driver_and_cursor, _worker_state, vision_determinism
from records import prompt_hash, tensor_save, value_hash
from replay import load_generation
from probe import fixed_response
from baseline_acceptance import role_masks_match
from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup


def test_native_scheduler_initial_lr_and_full_horizon():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([p], lr=5e-7)
    get_constant_schedule_with_warmup(optimizer, num_warmup_steps=0)
    assert optimizer.param_groups[0]["lr"] == 0  # Witness the native zero-first-step behavior.
    optimizer = torch.optim.AdamW([p], lr=5e-7)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=32)
    assert optimizer.param_groups[0]["lr"] == 5e-7
    p.grad = torch.ones_like(p)
    before = p.detach().clone()
    optimizer.step()
    assert not torch.equal(p, before)
    scheduler.step()
    import math
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-7 * (1 + math.cos(math.pi / 32)) / 2)


def test_adapter_keeps_native_role_chat_template():
    import os
    import subprocess
    code = ("from verl.workers.rollout.vllm_rollout.vllm_agent0_rollout_spmd import vLLMAgent0Rollout;"
            "import native_adapter; original=vLLMAgent0Rollout._get_template_tokens;"
            "native_adapter._patch_workers();"
            "assert vLLMAgent0Rollout._get_template_tokens is original, 'Native chat roles wrapped twice'")
    env = os.environ.copy()
    env.pop("BASELINE_RUN_DIR", None)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + str(ROOT / ".runtime/upstream/Agent0/Agent0-VL")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_vision_flash_attention_explicitly_forwards_native_deterministic_flag(monkeypatch):
    from importlib import import_module
    calls = []
    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return "native-result"
    for name in ("qwen2_vl", "qwen2_5_vl"):
        module = import_module(f"transformers.models.{name}.modeling_{name}")
        monkeypatch.setattr(module, "flash_attn_varlen_func", native)
    assert not any(vision_determinism().values())
    assert all(vision_determinism(enable=True).values())
    for name in ("qwen2_vl", "qwen2_5_vl"):
        module = import_module(f"transformers.models.{name}.modeling_{name}")
        assert module.flash_attn_varlen_func("q", "k", "v", deterministic=False) == "native-result"
    assert len(calls) == 2 and all(kwargs["deterministic"] is True for _, kwargs in calls)


def test_fixed_verifier_response_keeps_mrope_axes_and_visual_input():
    class Tokenizer:
        def encode(self, text, **kwargs):
            return [9, 10]
    position = torch.tensor([[[0, 1, 2], [0, 1, 2], [0, 1, 2]]])
    prompts = DataProto(batch=TensorDict({"input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[0, 1, 1]]), "position_ids": position}, batch_size=[1]),
        non_tensor_batch={"visual": np.array(["same-image"], dtype=object)})
    scoring, _ = fixed_response(prompts, Tokenizer())
    assert scoring.batch["position_ids"].tolist() == [[[0, 1, 2, 3, 4]] * 3]
    assert scoring.batch["input_ids"].tolist() == [[1, 2, 3, 9, 10]]
    assert scoring.batch["attention_mask"].tolist() == [[0, 1, 1, 1, 1]]
    assert scoring.non_tensor_batch["visual"].tolist() == ["same-image"]
    assert prompts.batch["input_ids"].tolist() == [[1, 2, 3]]


@pytest.mark.parametrize("damage", ["none", "trained_tool", "role_flag", "token", "attention_hole"])
def test_role_mask_rebuilt_from_semantic_roles_and_real_token_boundaries(damage):
    segments = np.empty(1, dtype=object)
    segments[0] = [{"role": "solver", "start": 0, "end": 1, "trainable": True, "token_ids": [10]},
                   {"role": "tool_observation", "start": 1, "end": 2, "trainable": False, "token_ids": [20]},
                   {"role": "verifier", "start": 2, "end": 3, "trainable": True, "token_ids": [30]}]
    data = DataProto(batch=TensorDict({"responses": torch.tensor([[10, 20, 30, 0]]),
        "multiturn_mask": torch.tensor([[True, False, True, False]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]])}, batch_size=[1]),
        non_tensor_batch={"baseline_segments": segments})
    if damage == "trained_tool":
        data.batch["multiturn_mask"][0, 1] = True
    elif damage == "role_flag":
        segments[0][1]["role"] = "verifier"
    elif damage == "token":
        data.batch["responses"][0, 1] = 99
    elif damage == "attention_hole":
        data.batch["attention_mask"][0, 2] = 0
        data.batch["attention_mask"][0, 4] = 1
    assert role_masks_match(data) is (damage == "none")


def data_with_answers(answers):
    count = len(answers)
    segments = np.empty(count, dtype=object)
    info = np.empty(count, dtype=object)
    for i, answer in enumerate(answers):
        segments[i] = [{"role": "solver", "text": f"\\boxed{{{answer}}}", "truncated": False}]
        info[i] = {"case_id": "task" + str(i // 4), "answer": "5"}
    return DataProto(batch=TensorDict({"attention_mask": torch.ones(count, 8, dtype=torch.long)}, batch_size=[count]),
                     non_tensor_batch={"extra_info": info, "baseline_segments": segments,
                                       "baseline_slot": np.arange(count) % 4})


def test_semantic_false_positive_and_hand_computed_whole_group():
    data = data_with_answers([9, 5, 8, 5, 5, 5, 5, 5])
    scores = torch.tensor([[0., 0., 0., v] for v in [0, .2, 1, .6, .2, .4, .6, .8]])
    before = scores.clone()
    revised, cert = false_positive(data, scores, [0, 1, 0, 1, 1, 1, 1, 1], step=8, alpha_out=1)
    assert torch.equal(scores, before) and cert["row"] == 0
    assert (revised != scores).nonzero().tolist() == [[0, 3]]
    group = np.array(["first"] * 4 + ["second"] * 4, dtype=object)
    clean, _ = compute_grpo_outcome_advantage(scores.clone(), torch.ones_like(scores), group)
    changed, _ = compute_grpo_outcome_advantage(revised.clone(), torch.ones_like(scores), group)
    returns = revised.sum(-1).tolist()[:4]
    expected = [(x - statistics.mean(returns)) / (statistics.stdev(returns) + 1e-6) for x in returns]
    assert torch.allclose(changed[:4, 0], torch.tensor(expected), atol=1e-6)
    assert torch.all(changed[:4, 0] != clean[:4, 0])
    assert torch.equal(changed[4:], clean[4:])


def test_missing_or_truncated_solver_answer_cannot_be_main_target():
    data = data_with_answers([9, 5, 5, 5])
    data.non_tensor_batch["baseline_segments"][0] = [{"role": "verifier", "text": "\\boxed{9}"}]
    rewards = torch.zeros(4, 4)
    revised, cert = false_positive(data, rewards, [0, 1, 1, 1], step=8, alpha_out=1)
    assert cert["status"] == "no_eligible_target" and torch.equal(revised, rewards)
    data.non_tensor_batch["baseline_segments"][0] = [{"role": "solver", "text": "\\boxed{9}", "truncated": True}]
    _, cert = false_positive(data, rewards, [0, 1, 1, 1], step=8, alpha_out=1)
    assert cert["status"] == "no_eligible_target"


def test_delta0_control_preserves_all_native_rewards_and_advantages():
    data = data_with_answers([9, 5, 8, 5])
    rewards = torch.tensor([[0., 0., 0., r] for r in (.2, 1., .4, .8)])
    unchanged, cert = false_positive(data, rewards, [0, 1, 0, 1], step=8, alpha_out=1, delta=0)
    assert cert["status"] == "no_op" and cert["delta_return"] == 0
    assert torch.equal(rewards, unchanged)
    ids = np.array(["group"] * 4, dtype=object)
    clean, _ = compute_grpo_outcome_advantage(rewards, torch.ones_like(rewards), ids)
    control, _ = compute_grpo_outcome_advantage(unchanged, torch.ones_like(rewards), ids)
    assert torch.equal(clean, control)


def test_delta0_preserves_negative_zero_bytes():
    data = data_with_answers([9, 5, 8, 5])
    rewards = torch.zeros(4, 4)
    rewards[0, -1] = -0.0
    unchanged, cert = false_positive(data, rewards, [0, 1, 0, 1], step=8, alpha_out=1, delta=0)
    assert cert["status"] == "no_op" and value_hash(rewards) == value_hash(unchanged)


def test_update_comparison_rejects_equal_values_with_changed_tensor_dtype(tmp_path):
    import copy
    from baseline_acceptance import compare
    a, b = recorded_output(), recorded_output()
    b.batch["multiturn_mask"] = b.batch["multiturn_mask"].to(torch.int64)
    assert torch.equal(a.batch["multiturn_mask"], b.batch["multiturn_mask"])
    worker = [{"model": {}, "buffers": {}, "optimizer": {}, "optimizer_groups": [], "scheduler": {}, "rng_hash": "R"}]
    for name, group in (("a", a), ("b", b)):
        run = tmp_path / name
        atomic_json(run / "manifest.json", {"run_id": name, "signatures": {"learning": "same"}})
        tensor_save(group, run / "raw/training_group_step_000001.pt")
        atomic_json(run / "checkpoints/global_step_1/baseline_worker_state.json", copy.deepcopy(worker))
    result = compare(tmp_path / "a", tmp_path / "b", 1)
    assert not result["passed"] and not result["tensor_exact"]["multiturn_mask"]


def test_driver_rng_isolation_has_same_next_draws():
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    state = rng_state()
    expected = (random.random(), np.random.random(), torch.rand(2))
    restore_rng(state)
    eval_state = rng_state()
    for _ in range(11):
        random.random(); np.random.random(); torch.rand(7)
    restore_rng(eval_state)
    assert random.random() == expected[0] and np.random.random() == expected[1]
    assert torch.equal(torch.rand(2), expected[2])


def test_complete_state_fingerprint_includes_persistent_buffers_and_rejects_nan(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    module = torch.nn.Linear(1, 1)
    module.register_buffer("running_value", torch.tensor([2.]))
    module.register_buffer("derived_cache", torch.tensor([3.]), persistent=False)
    optimizer = torch.optim.AdamW(module.parameters(), lr=.1)
    worker = SimpleNamespace(actor=SimpleNamespace(actor_module=module, actor_optimizer=optimizer),
                             actor_lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer, 1))
    first = _worker_state(worker)
    assert set(first["buffers"]) == {"running_value"}
    module.running_value.add_(1)
    assert _worker_state(worker)["buffers"] != first["buffers"]
    module.running_value.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="buffer"):
        _worker_state(worker)


@pytest.mark.parametrize("completed_batches", [0, 7])
def test_lazy_cursor_restoration_keeps_driver_stream_and_skips_completed_batches(tmp_path, completed_batches):
    from torchdata.stateful_dataloader import StatefulDataLoader
    from torch.utils.data import TensorDataset
    torch.manual_seed(20262001)
    dataset = TensorDataset(torch.arange(128))
    original = StatefulDataLoader(dataset, batch_size=4, num_workers=2, shuffle=False)
    iterator = iter(original)
    for _ in range(completed_batches):
        next(iterator)
    cursor, driver = original.state_dict(), rng_state()
    expected_draws = torch.rand(3)
    tensor_save(cursor, tmp_path / "data.pt")
    tensor_save(driver, tmp_path / "baseline_driver_rng.pt")
    resumed = StatefulDataLoader(dataset, batch_size=4, num_workers=2, shuffle=False)
    resumed.load_state_dict(cursor)
    receipt = restore_driver_and_cursor(resumed, tmp_path)
    assert receipt["driver_rng_sha256"] == value_hash(driver) == value_hash(rng_state())
    assert receipt["dataloader_sha256"] == value_hash(cursor)
    assert next(iter(resumed))[0].tolist() == list(range(4 * completed_batches, 4 * completed_batches + 4))
    assert torch.equal(torch.rand(3), expected_draws)


def recorded_output():
    return DataProto(batch=TensorDict({"prompts": torch.tensor([[1]]), "responses": torch.tensor([[3, 4]]),
        "input_ids": torch.tensor([[1, 3, 4]]), "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "position_ids": torch.tensor([[0, 1, 2]]), "multiturn_mask": torch.tensor([[True, True]])}, batch_size=[1]),
        non_tensor_batch={"baseline_case_id": np.array(["case"], dtype=object), "baseline_slot": np.array([0]),
            "baseline_segments": np.array([["recorded"]], dtype=object), "step_data": np.array([["step"]], dtype=object),
            "verify_probs": np.array([[.9]]), "num_steps": np.array([1]), "num_repairs": np.array([0]),
            "final_answers": np.array(["3"], dtype=object), "final_generation_step": np.array([0]),
            "multi_modal_inputs": np.array(["same-visual-input"], dtype=object)})


def test_visual_replay_rejects_wrong_image_even_when_text_identical(tmp_path):
    prompts = DataProto(batch=TensorDict({"input_ids": torch.tensor([[1, 2]])}, batch_size=[1]),
                         non_tensor_batch={"image": np.array([np.zeros((2, 2))], dtype=object)})
    output = recorded_output()
    tensor_save({"step": 1, "learning_signature": "same", "prompt_hash": prompt_hash(prompts),
                 "output": output}, tmp_path / "raw/generation_step_000001.pt")
    tensor_save(output, tmp_path / "raw/training_group_step_000001.pt")
    atomic_json(tmp_path / "raw/group_step_000001.json", {"payload_hash": prompt_hash(output)})
    atomic_json(tmp_path / "raw/completed_step_000001.json", {"learning_signature": "same",
        "group_sha256": file_digest(tmp_path / "raw/group_step_000001.json")})
    assert torch.equal(load_generation(tmp_path, step=1, learning="same", prompts=prompts).batch["responses"], output.batch["responses"])
    prompts.non_tensor_batch["image"] = np.array([np.ones((2, 2))], dtype=object)
    with pytest.raises(ValueError, match="visual"):
        load_generation(tmp_path, step=1, learning="same", prompts=prompts)


@pytest.mark.parametrize("damage", ["none", "counter_value", "counter_float"])
def test_replay_matches_native_reward_counter_cast_without_accepting_other_changes(tmp_path, damage):
    import copy
    prompts = DataProto(batch=TensorDict({"input_ids": torch.tensor([[1]])}, batch_size=[1]))
    output = recorded_output()
    for key in ("num_steps", "num_repairs"):
        output.non_tensor_batch[key] = output.non_tensor_batch[key].astype(np.int32)
    group = copy.deepcopy(output)
    for key in ("num_steps", "num_repairs"):
        group.non_tensor_batch[key] = group.non_tensor_batch[key].astype(np.int64)
    tensor_save(group, tmp_path / "raw/training_group_step_000001.pt")
    atomic_json(tmp_path / "raw/group_step_000001.json", {"payload_hash": prompt_hash(group)})
    atomic_json(tmp_path / "raw/completed_step_000001.json", {"learning_signature": "same",
        "group_sha256": file_digest(tmp_path / "raw/group_step_000001.json")})
    if damage == "counter_value":
        output.non_tensor_batch["num_repairs"][0] = 1
    elif damage == "counter_float":
        output.non_tensor_batch["num_steps"] = output.non_tensor_batch["num_steps"].astype(np.float64)
    tensor_save({"step": 1, "learning_signature": "same", "prompt_hash": prompt_hash(prompts),
        "output": output}, tmp_path / "raw/generation_step_000001.pt")
    if damage == "none":
        assert len(load_generation(tmp_path, step=1, learning="same", prompts=prompts)) == 1
    else:
        with pytest.raises(ValueError, match="membership"):
            load_generation(tmp_path, step=1, learning="same", prompts=prompts)


@pytest.mark.parametrize("damage", ["mask", "mask_dtype", "group_slot", "missing_mask", "missing_member", "missing_visual"])
def test_replay_rejects_modified_mask_or_member(tmp_path, damage):
    prompts = DataProto(batch=TensorDict({"input_ids": torch.tensor([[1]])}, batch_size=[1]),
        non_tensor_batch={"multi_modal_inputs": np.array(["same-visual-input"], dtype=object)})
    output = recorded_output()
    tensor_save(output, tmp_path / "raw/training_group_step_000001.pt")
    atomic_json(tmp_path / "raw/group_step_000001.json", {"payload_hash": prompt_hash(output)})
    atomic_json(tmp_path / "raw/completed_step_000001.json", {"learning_signature": "same",
        "group_sha256": file_digest(tmp_path / "raw/group_step_000001.json")})
    if damage == "mask":
        output.batch["multiturn_mask"][0, 0] = False
    elif damage == "mask_dtype":
        output.batch["multiturn_mask"] = output.batch["multiturn_mask"].double()
    elif damage == "missing_mask":
        del output.batch["multiturn_mask"]
    elif damage == "missing_member":
        del output.non_tensor_batch["baseline_slot"]
    elif damage == "missing_visual":
        del output.non_tensor_batch["multi_modal_inputs"]
    else:
        output.non_tensor_batch["baseline_slot"][0] = 1
    tensor_save({"step": 1, "learning_signature": "same", "prompt_hash": prompt_hash(prompts),
        "output": output}, tmp_path / "raw/generation_step_000001.pt")
    with pytest.raises(ValueError, match="mask|membership"):
        load_generation(tmp_path, step=1, learning="same", prompts=prompts)
