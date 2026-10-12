"""Single semantic terminal false-positive without changing native rollout."""
from __future__ import annotations

from evaluate import solver_answer, integer_answer
from pilot_utils import digest
from records import trajectory_id


def false_positive(data, rewards, native_acc, *, step: int, alpha_out: float, delta: int = 1) -> tuple:
    if alpha_out != 1.0:
        raise ValueError("The proposed main intervention requires alpha_out=1")
    if type(delta) is not int or delta not in (0, 1):
        raise ValueError("Use terminal false-positive delta1 or its declared delta0 control")
    eligible = []
    for row, (info, segments) in enumerate(zip(data.non_tensor_batch["extra_info"],
                                              data.non_tensor_batch["baseline_segments"])):
        answer = solver_answer(segments)
        if answer is not None and answer != integer_answer(info["answer"]) and native_acc[row] == 0:
            if not any(segment.get("truncated", False) for segment in segments):
                identity = trajectory_id(step, info["case_id"], int(data.non_tensor_batch["baseline_slot"][row]))
                eligible.append((identity, row))
    if not eligible:
        return rewards.clone(), {"status": "no_eligible_target", "step": step, "requested_delta": delta}
    identity, row = min(eligible)
    positions = data.batch["attention_mask"][row, -rewards.shape[-1]:].bool().nonzero().flatten()
    if not positions.numel():
        raise ValueError("Cannot revise empty trajectory")
    revised = rewards.clone()
    if delta:
        revised[row, positions[-1]] += alpha_out * delta
    certificate = {"status": "applied" if delta else "no_op", "step": step, "trajectory_id": identity, "row": row,
                   "component": "terminal_outcome", "old_outcome": 0, "new_outcome": delta,
                   "delta_return": float(delta), "visible_at": "after final unit fixed, before GRPO",
                   "selection": "first stable semantic-error identity; no future performance used"}
    certificate["certificate_signature"] = digest(certificate)
    return revised, certificate
