"""Case-level arithmetic evaluation using explicit Solver output only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

from pilot_utils import atomic_json, read_json


def integer_answer(value) -> int | None:
    if value is None or not re.fullmatch(r"[+-]?\d+", str(value).strip(), flags=re.ASCII):
        return None
    return int(str(value).strip())


def solver_answer(segments: list[dict]) -> int | None:
    answer = None
    for segment in segments:
        if segment["role"] not in ("solver", "regenerate"):
            continue
        matches = re.findall(r"\\boxed\{([^{}]*)\}|FINAL_ANSWER:\s*([^\s]+)",
                             segment["text"], flags=re.ASCII)
        if matches:
            answer = integer_answer(next((x for x in matches[-1] if x), ""))
    return answer


def case_result(info: dict, segments: list[dict], native_accuracy: float,
                step_data: list[dict]) -> dict:
    prediction = solver_answer(segments)
    truth = integer_answer(info["answer"])
    if truth is None:
        raise ValueError("Arithmetic checker requires an integer ground truth")
    return {"case_id": info["case_id"], "image_sha256": info["image_sha256"],
            "scene_id": info["scene_id"], "family": info["family"],
            "explicit_answer": prediction, "ground_truth": truth,
            "explicit_answer_present": prediction is not None,
            "correct": int(prediction is not None and prediction == truth),
            "native_accuracy": float(native_accuracy),
            "native_fallback_used": prediction is None,
            "tool_calls": sum(bool(s.get("tool_used")) for s in step_data),
            "tool_successes": sum(bool(s.get("tool_success")) for s in step_data),
            "verified_steps": sum(bool(s.get("verified")) for s in step_data),
            "repair_steps": sum(bool(s.get("was_repaired")) for s in step_data),
            "truncated": any(s.get("truncated", False) for s in segments)}


def aggregate(cases: list[dict]) -> dict:
    if not cases or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Evaluation requires nonempty, unique case IDs")
    families = {}
    for family in sorted({str(c["family"]) for c in cases}):
        rows = [c for c in cases if str(c["family"]) == family]
        families[family] = {"cases": len(rows), "correct": sum(c["correct"] for c in rows),
                            "accuracy": sum(c["correct"] for c in rows) / len(rows)}
    return {"cases": len(cases), "correct": sum(c["correct"] for c in cases),
            "accuracy": sum(c["correct"] for c in cases) / len(cases),
            "explicit_answer_rate": sum(c["explicit_answer_present"] for c in cases) / len(cases),
            "native_accuracy": sum(c["native_accuracy"] for c in cases) / len(cases),
            "truncation_rate": sum(c["truncated"] for c in cases) / len(cases),
            "families": families}


def write_report(raw: Path, output: Path) -> dict:
    receipt = read_json(raw)
    summary = aggregate(receipt["cases"])
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "evaluation.json", summary)
    with (output / "cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(receipt["cases"][0]))
        writer.writeheader()
        writer.writerows(receipt["cases"])
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_report(args.raw, args.output), ensure_ascii=False))
