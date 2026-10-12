"""Separate logical rollout requests from duplicated tensor-parallel records."""
from __future__ import annotations

from pathlib import Path

from pilot_utils import digest, read_json


def logical_role_counts(run: Path) -> dict:
    tp = read_json(run / "manifest.json")["overrides"]["actor_rollout_ref.rollout.tensor_model_parallel_size"]
    calls = {}
    for path in sorted((run / "raw/roles").glob("*.json")):
        record = read_json(path)
        counter = int(path.stem.rsplit("_call_", 1)[1])
        group = record["rank"] // tp
        key = (record["phase"], record["step"], record["role"], record["inner_step"], group, counter)
        sha = digest(record["requests"])
        if key in calls:
            item = calls[key]
            if item["sha256"] != sha or record["rank"] in item["ranks"]:
                raise ValueError("TP request records disagree or repeat a rank")
            item["ranks"].add(record["rank"])
        else:
            calls[key] = {"sha256": sha, "ranks": {record["rank"]}, "requests": record["requests"]}
    totals = {}
    for key, call in calls.items():
        phase, _, role, _, group, _ = key
        if call["ranks"] != set(range(group * tp, (group + 1) * tp)):
            raise ValueError("Incomplete TP records; logical counts cannot be certified")
        part = totals.setdefault(phase + "/" + role, {"logical_batch_calls": 0, "logical_requests": 0, "generated_tokens": 0})
        part["logical_batch_calls"] += 1
        part["logical_requests"] += len(call["requests"])
        part["generated_tokens"] += sum(len(request["output_token_ids"]) for request in call["requests"])
    return {"phase_role": totals, "tp_size": tp,
            "basis": "Exact request/response records deduplicated within native contiguous TP groups and per-worker call occurrence; repeated evaluations remain distinct. Worker wall times are not summed as elapsed time."}
