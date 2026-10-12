"""Rebuild Gate1 statistics from all five seeds and verified immutable raw evidence."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile

from evaluate import aggregate
from gate1_contract import paired_interval
from gate1_evidence import SEEDS, evidence, finite, physical, validate_freeze, verify_evidence
from gate1_state import certified_snapshot
from gate1_validation import record_run, validate_unit
from pilot_utils import VERSION, atomic_json, digest, file_digest, read_json
from records import fsync_directory
from run_gate1_sequence import eligibility, paired_proof


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError("Cannot write an empty result table")
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def auxiliary_diagnostics(evaluation: dict, seed: int, step: int, branch: str) -> tuple[dict, list[dict]]:
    solver = aggregate(evaluation["cases"])
    verifier = evaluation["verifier_probes"]
    rows = verifier["cases"]
    wrong = [row for row in rows if row["expected_correct"] is False]
    correct = [row for row in rows if row["expected_correct"] is True]
    if (not wrong or not correct or len({row["probe_id"] for row in rows}) != len(rows) or
            len(wrong) + len(correct) != len(rows) or not finite(verifier) or
            any(type(row["parsed"]) is not bool or type(row["accepted"]) is not bool for row in rows)):
        raise ValueError("Verifier auxiliary cases are missing, duplicated or invalid")
    calculated = {"cases": len(rows), "parse_rate": sum(row["parsed"] for row in rows) / len(rows),
                  "incorrect_acceptance_rate": sum(row["accepted"] for row in wrong) / len(wrong),
                  "correct_acceptance_rate": sum(row["accepted"] for row in correct) / len(correct)}
    if any(verifier["summary"][key] != value for key, value in calculated.items()):
        raise ValueError("Verifier auxiliary summary differs from its raw fixed cases")
    result = {"seed": seed, "step": step, "branch": branch, "solver_cases": solver["cases"],
              "solver_accuracy": solver["accuracy"], "solver_explicit_answer_rate": solver["explicit_answer_rate"],
              "solver_truncation_rate": solver["truncation_rate"], "verifier_split": verifier["split"],
              "verifier_cases": calculated["cases"], "verifier_parse_rate": calculated["parse_rate"],
              "verifier_correct_acceptance_rate": calculated["correct_acceptance_rate"],
              "repeated_false_positive_rate": calculated["incorrect_acceptance_rate"]}
    for key in ["tool_calls", "tool_successes", "verified_steps", "repair_steps"]:
        result[key] = sum(row[key] for row in evaluation["cases"])
    families = [{"seed": seed, "step": step, "branch": branch, "family": family, **values}
                for family, values in solver["families"].items()]
    return result, families


def export(sequence: Path):
    sequence = physical(sequence, VERSION / "outputs")
    status = read_json(sequence / "status.json")
    manifest = read_json(sequence / "manifest.json")
    freeze_path = physical(Path(manifest["freeze_path"]))
    freeze = validate_freeze(freeze_path)
    if (status["status"] != "completed" or file_digest(freeze_path) != manifest["freeze_sha256"] or
            manifest["seeds"] != SEEDS or manifest["horizon"] != 32 or manifest["k"] != 8):
        raise ValueError("Require the complete predeclared five-seed sequence and unchanged freeze")
    assets_path = physical(Path(manifest["inputs"]["assets"]))
    reference_directory = physical(Path(manifest["inputs"]["references"]))
    if (file_digest(assets_path) != manifest["input_sha256"]["assets"] or
            file_digest(reference_directory / "manifest.json") != manifest["input_sha256"]["references"]):
        raise ValueError("Source import or historical reference manifest changed")
    assets = read_json(assets_path)
    references = read_json(reference_directory / "manifest.json")
    if (assets["manifest_signature"] != digest({k: v for k, v in assets.items() if k != "manifest_signature"}) or
            references["manifest_signature"] != digest({k: v for k, v in references.items() if k != "manifest_signature"})):
        raise ValueError("Unsigned source/reference provenance")
    for name, info in assets["files"].items():
        if evidence(physical(assets_path.parent / name))["sha256"] != info["sha256"]:
            raise ValueError("Imported evidence was corrupted")
    blocks = status["completed_seed_blocks"]
    if [block["seed"] for block in blocks] != SEEDS or any(type(block["valid"]) is not bool for block in blocks):
        raise ValueError("Missing/selected/duplicate/extra planned seed")
    pairs, curves, case_rows, effects, evidence_rows, diagnostic_rows, family_rows = [], [], [], [], [], [], []
    for block in blocks:
        seed = block["seed"]
        source = physical(Path(block["source"]))
        source_manifest = read_json(source / "manifest.json")
        if (source_manifest["seed"] != seed or read_json(source / "status.json") != {"status": "completed", "returncode": 0} or
                source_manifest["checkpoint_plan"]["end"] != 32 or
                digest(source_manifest["learning"]) != source_manifest["signatures"]["learning"]):
            raise ValueError("Seed source identity differs")
        source_proof_path = sequence / f"raw/source_seed_{seed}.json"
        source_proof = read_json(source_proof_path)
        if source_proof["seed"] != seed or physical(Path(source_proof["source"])) != source:
            raise ValueError("Source acceptance proof belongs to another seed/path")
        if source_proof["kind"] == "verified_import":
            if source_proof["import_manifest_sha256"] != manifest["input_sha256"]["assets"] or str(seed) not in assets["histories"]:
                raise ValueError("Historical source import differs")
            full_source = Path(references["references"][f"{seed}:32"]["source"])
        elif source_proof["kind"] == "accepted_preparation":
            accepted_path = physical(Path(source_proof["acceptance"]))
            accepted = read_json(accepted_path)
            verify_evidence(accepted["validation"]["evidence"])
            actual_source = physical(Path(source_proof["actual_source"]))
            if accepted["run_id"] != actual_source.name or accepted["validation"]["origin_manifest"]["seed"] != seed:
                raise ValueError("Prepared source acceptance belongs to another attempt/seed")
            full_source = actual_source / "checkpoints/global_step_32"
            evidence_rows.extend([evidence(accepted_path), *accepted["validation"]["evidence"]])
        else:
            raise ValueError("Unknown source provenance")
        source_certificate = certified_snapshot(full_source)
        if source_certificate["run_id"] != source_manifest["run_id"] or source_certificate["step"] != 32:
            raise ValueError("Planned source S32 is not retained as its actual full state")
        evidence_rows.extend([evidence(source_proof_path), evidence(full_source / "baseline_complete.json")])
        revision_path = sequence / f"raw/eligibility_{seed}.json"
        revision = read_json(revision_path)
        if revision != eligibility(source):
            raise ValueError("Recorded first eligible target differs from actual U8")
        evidence_rows.extend([evidence(revision_path), evidence(source / "manifest.json")])
        if not block["valid"]:
            if block.get("status") != "no_eligible_target" or revision["status"] != "no_eligible_target":
                raise ValueError("Invalid seed omitted without its actual eligibility receipt")
            for step in range(1, 33):
                evidence_rows.extend(validate_unit(source, step, expected_injected=False))
            pairs.append({"seed": seed, "status": "no_eligible_target", "D_score_T": None, "floor_T": None,
                          "corrected_D_T": None, "Q_s": None, "P_clean_T": None, "P_corrupt_T": None, "Delta_P_T": None})
            continue
        if revision["status"] != "applied":
            raise ValueError("Valid paired seed has no eligible intervention")
        runs = {key: physical(Path(value), VERSION / "outputs") for key, value in block["branches"].items()}
        if set(runs) != {"clean", "corrupt", "repeat"}:
            raise ValueError("Missing Gate1 branch")
        prefix = physical(Path(block["prefix"]))
        reference = physical(Path(references["references"][f"{seed}:32"]["path"])) if str(seed) in assets["histories"] else None
        raw_proof = paired_proof(sequence, seed, source, prefix, runs, revision, manifest["freeze_sha256"], reference)
        proof_path = physical(Path(block["proof"]))
        if proof_path != sequence / f"raw/paired_seed_{seed}.json":
            raise ValueError("Paired acceptance proof belongs to another sequence/seed")
        stored = read_json(proof_path)
        verify_evidence(stored["evidence"])
        if stored != raw_proof:
            raise ValueError("Paired proof differs from rebuilt actual branch/source evidence")
        evidence_rows.extend([evidence(proof_path), *raw_proof["evidence"]])
        values = {}
        for point in raw_proof["curve"]:
            step = point["step"]
            accuracy = {}
            for branch, run in runs.items():
                evaluation = read_json(Path(point["evaluation"])) if step == 0 else read_json(
                    record_run(run, step) / f"raw/evaluation_step_{step:06d}.json")
                accuracy[branch] = aggregate(evaluation["cases"])["accuracy"]
                auxiliary, families = auxiliary_diagnostics(evaluation, seed, step, branch)
                diagnostic_rows.append(auxiliary)
                family_rows.extend(families)
                case_rows.extend({"seed": seed, "step": step, "branch": branch,
                    "provenance": point.get("provenance", "paired fixed branches"), **case} for case in evaluation["cases"])
            row = {"seed": seed, "step": step, "D_score": point["D_score"], "D_solver": point["roles"]["solver"],
                   "D_verifier": point["roles"]["verifier"], "floor": point["floor"],
                   "corrected_D": point["corrected_D"], "Q_s": point["corrected_D"] - freeze["epsilon_D"],
                   "provenance": point.get("provenance", "paired fixed branches"),
                   "P_clean": accuracy["clean"], "P_corrupt": accuracy["corrupt"], "P_repeat": accuracy["repeat"]}
            curves.append(row)
            values[step] = row
        final = values[32]
        effects.append(final["corrected_D"])
        pairs.append({"seed": seed, "status": "valid", "D_score_T": final["D_score"], "floor_T": final["floor"],
                      "corrected_D_T": final["corrected_D"], "Q_s": final["Q_s"], "P_clean_T": final["P_clean"],
                      "P_corrupt_T": final["P_corrupt"], "Delta_P_T": final["P_clean"] - final["P_corrupt"]})
    inference = paired_interval(effects, freeze["epsilon_D"]) if effects else {
        "n": 0, "mean": None, "lower": None, "upper": None, "epsilon_D": freeze["epsilon_D"],
        "decision": "uncertain", "reason": "no eligible target in any planned seed"}
    for key in ["mean", "lower", "upper"]:
        if inference.get(key) is not None:
            inference[key] -= freeze["epsilon_D"]
    inference["quantity"] = "Q_s = D(Clean,Corrupt) - D(Clean,Clean-repeat) - epsilon_D; not clipped"
    output = sequence / "processed"
    output.mkdir(exist_ok=True)
    write_csv(output / "paired_seeds.csv", pairs)
    if curves:
        write_csv(output / "score_curves.csv", curves)
    if case_rows:
        write_csv(output / "heldout_cases.csv", case_rows)
    if diagnostic_rows:
        write_csv(output / "auxiliary_diagnostics.csv", diagnostic_rows)
        write_csv(output / "heldout_families.csv", family_rows)
    result = {"planned_seeds": SEEDS, "valid_seeds": len(effects), "primary": inference, "horizon": 32, "k": 8,
              "freeze_sha256": manifest["freeze_sha256"], "source_sha256": file_digest(Path(__file__)),
              "raw_evidence": evidence_rows, "statistics_unit": "independent paired training seed",
              "auxiliary_repeated_error": "Verifier acceptance of independent fixed wrong-answer contexts measures the injected false-positive type",
              "interpretation": "Fixed-probe/native-reanchor persistence only; no full-policy KL, capability-loss or SCOR claim"}
    atomic_json(output / "result_summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", type=Path)
    args = parser.parse_args()
    print(export(args.sequence))
