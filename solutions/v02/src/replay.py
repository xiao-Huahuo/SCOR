"""Validate full visual input identity before native reanchor replay."""
from __future__ import annotations

from pathlib import Path

from pilot_utils import file_digest, read_json
from records import prompt_hash, value_hash


def load_generation(source: Path, *, step: int, learning: str, prompts):
    import torch
    # Only own trusted project outputs are accepted by the launcher.
    payload = torch.load(source / f"raw/generation_step_{step:06d}.pt",
                         map_location="cpu", weights_only=False)
    if payload["learning_signature"] != learning or payload["step"] != step:
        raise ValueError("History learning identity or update position differs")
    if payload["prompt_hash"] != prompt_hash(prompts):
        raise ValueError("History visual input, task order or context differs")
    output = payload["output"]
    tensors = {"prompts", "responses", "input_ids", "attention_mask", "position_ids", "multiturn_mask"}
    metadata = {"baseline_case_id", "baseline_slot", "baseline_segments", "step_data", "verify_probs",
                "num_steps", "num_repairs", "final_answers", "final_generation_step"}
    if "multi_modal_inputs" in prompts.non_tensor_batch:
        metadata.add("multi_modal_inputs")
    if set(output.batch.keys()) != tensors or not metadata.issubset(output.non_tensor_batch):
        raise ValueError("History generation schema omits inputs, masks or native role/group records")
    completed = read_json(source / f"raw/completed_step_{step:06d}.json")
    group_path = source / f"raw/group_step_{step:06d}.json"
    if completed["learning_signature"] != learning or file_digest(group_path) != completed["group_sha256"]:
        raise ValueError("History completion or group receipt identity differs")
    group = torch.load(source / f"raw/training_group_step_{step:06d}.pt", map_location="cpu", weights_only=False)
    if prompt_hash(group) != read_json(group_path)["payload_hash"]:
        raise ValueError("History training group content differs from its receipt")
    if any(key not in group.batch or value_hash(value) != value_hash(group.batch[key]) for key, value in output.batch.items()):
        raise ValueError("History token, image position or mask differs from committed unit")
    for key, value in output.non_tensor_batch.items():
        if key not in group.non_tensor_batch:
            raise ValueError("History visual metadata, role record or group membership differs")
        target = group.non_tensor_batch[key]
        # Native reward_extra_info replaces these int32 rollout counters with
        # np.array(Python-int-list), hence int64, before the updater is recorded.
        if (key in ("num_steps", "num_repairs") and
                str(getattr(value, "dtype", None)) == "int32" and
                str(getattr(target, "dtype", None)) == "int64"):
            value = value.astype(target.dtype)
        if value_hash(value) != value_hash(target):
            raise ValueError("History visual metadata, role record or group membership differs")
    return payload["output"]
