"""Prepare immutable scene-disjoint native baseline inputs and audit probes."""
from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont
import pyarrow as pa
import pyarrow.parquet as pq

from pilot_utils import SRC, VERSION, atomic_json, digest, file_digest, read_json

DEFAULT_COUNTS = {"train": 1024, "calibration": 64, "probe": 128, "evaluation": 256}
SYSTEM = (
    "You are a visual reasoning agent. Follow the role explicitly requested in the latest instruction. "
    "Initially you are the Solver. Read the numeric labels from the image. "
    "Only in the Solver role: before any tool feedback exists, output fenced Python code that prints the result. "
    "Only in the Solver role: after successful execution, do not repeat the code; return exactly \\boxed{integer}. "
    "The integer must use decimal digits with an optional minus sign, without commas or mathematical expressions. "
    "Solver format example: for a result of 7, the entire answer is \\boxed{7}, including the backslash and braces. "
    "If a repair changes the calculation, execute the changed code before giving its final result. "
    "In the Verifier role, output the requested verification JSON only, never Python or a Solver answer. "
    "In the Self-Repair role, output the requested PATCH or NO_CHANGE JSON only."
)


def make_row(index: int, split: str, rng: random.Random) -> tuple[dict, dict]:
    import hashlib
    values = rng.sample(range(100, 1000), 3)
    family = index % 3
    smallest, middle, largest = sorted(values)
    answers = (sum(values), values[0] * values[1] * values[2], largest - smallest - middle)
    questions = (
        "What is the sum of the values of all three bars (red, blue and green)?",
        "What is the product of the values of all three bars (red, blue and green)?",
        "What is the largest bar value minus the sum of the other two bar values?",
    )
    image = Image.new("RGB", (512, 512), "white")
    draw, font = ImageDraw.Draw(image), ImageFont.load_default(size=24)
    draw.text((28, 20), "VALUES", fill="black", font=font)
    for i, (value, color) in enumerate(zip(values, ("red", "blue", "green"))):
        x, height = 65 + i * 145, int(value * .28)
        draw.rectangle((x, 410 - height, x + 80, 410), fill=color)
        draw.text((x + 15, 420), color, fill="black", font=font)
        draw.text((x + 18, 370 - height), str(value), fill="black", font=font)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    answer, image_bytes = str(answers[family]), buffer.getvalue()
    sample_id = digest({"values": values, "family": family})
    prompt = ("<image>\n" + questions[family] + " Read the numeric labels from the image. "
              "Use a fenced Python code block to calculate and print the result. "
              "When giving the final answer, write it as \\boxed{number} using decimal digits only.")
    row = {"data_source": "scor_visual_arithmetic_baseline", "prompt": [{"role": "user", "content": prompt}],
           "images": [{"bytes": image_bytes, "path": None}], "ability": "visual_arithmetic",
           "reward_model": {"style": "rule", "ground_truth": answer},
           "extra_info": {"index": index, "split": split, "sample_id": sample_id}}
    receipt = {"index": index, "split": split, "sample_id": sample_id, "scene_id": digest(values),
               "values": values, "family": family, "answer": answer,
               "image_sha256": hashlib.sha256(image_bytes).hexdigest()}
    return row, receipt


def prepare(destination: Path, seed: int = 20261011, counts: dict | None = None) -> dict:
    counts = dict(DEFAULT_COUNTS if counts is None else counts)
    if set(counts) != set(DEFAULT_COUNTS) or any(type(v) is not int or v <= 0 for v in counts.values()):
        raise ValueError("Specify four positive split sizes")
    if sum(counts.values()) > 900 * 899 * 898:
        raise ValueError("Requested split sizes exceed the distinct scene space")
    if destination.exists():
        raise FileExistsError("Use a new immutable dataset directory")
    destination.mkdir(parents=True)
    rng, seen, splits, receipts, probes = random.Random(seed), set(), {}, [], []
    verifier_splits = {}
    for split, count in counts.items():
        rows = []
        verifier_rows = []
        while len(rows) < count:
            row, receipt = make_row(len(rows), split, rng)
            if receipt["scene_id"] in seen:
                continue
            seen.add(receipt["scene_id"])
            receipt["case_id"] = digest([receipt["scene_id"], receipt["family"], row["prompt"][-1]["content"]])
            row["data_source"] = "scor_visual_arithmetic_baseline"
            row["prompt"].insert(0, {"role": "system", "content": SYSTEM})
            row["extra_info"].update({k: receipt[k] for k in
                                     ("case_id", "scene_id", "image_sha256", "family", "answer")})
            rows.append(row)
            receipts.append(receipt)
            if split in ("calibration", "probe"):
                import copy
                for correct in (True, False):
                    candidate = int(receipt["answer"]) + (0 if correct else 1)
                    probe_row = copy.deepcopy(row)
                    question = row["prompt"][-1]["content"].split(" Read the numeric labels from the image.")[0]
                    probe_row["prompt"][0]["content"] = (
                        "You are the Verifier for a visual arithmetic candidate. Your only task is to judge the supplied candidate "
                        "using the numeric labels in the image. Return the requested JSON. Never write Python or act as the Solver."
                    )
                    probe_row["prompt"][-1]["content"] = (
                        question + f"\nAs Verifier, evaluate this fixed candidate answer: \\boxed{{{candidate}}}. "
                        'Output exactly one JSON line: {"step_index":1,"score":<-1 to 1>,"confidence":<0 to 1>,"critique":"brief","tool_check":false}.'
                    )
                    probe_row["extra_info"].update({"base_case_id": receipt["case_id"],
                        "case_id": receipt["case_id"] + (":correct" if correct else ":incorrect"),
                        "expected_correct": correct})
                    verifier_rows.append(probe_row)
            if split == "probe":
                for correct in (True, False):
                    candidate = int(receipt["answer"]) + (0 if correct else 1)
                    probes.append({"case_id": receipt["case_id"], "image_sha256": receipt["image_sha256"],
                                   "candidate_text": f"The final answer is \\boxed{{{candidate}}}.",
                                   "expected_correct": correct, "label_visible_to_model": False})
        path = destination / f"{split}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
        splits[split] = {"rows": count, "file": path.name, "sha256": file_digest(path)}
        if verifier_rows:
            probe_path = destination / f"{split}_verifier.parquet"
            pq.write_table(pa.Table.from_pylist(verifier_rows), probe_path, compression="zstd")
            verifier_splits[split] = {"file": probe_path.name, "rows": len(verifier_rows), "sha256": file_digest(probe_path)}
        if split == "calibration":
            Image.open(BytesIO(rows[0]["images"][0]["bytes"])).save(destination / "preview.png")
    atomic_json(destination / "verifier_probes.json", probes, immutable=True)
    sources = {name: file_digest(SRC / name) for name in
               ("prepare_baseline_data.py", "evaluate.py")}
    manifest = {"kind": "controlled synthetic visual mechanism tasks; not paper benchmark",
                "seed": seed, "splits": splits, "samples": receipts,
                "verifier_splits": verifier_splits,
                "sources": sources, "checker": "explicit integer Solver answer",
                "task_definitions": {"0": "red + blue + green", "1": "red * blue * green",
                                     "2": "largest - (smallest + middle)"},
                "scene_definition": {"value_range_inclusive": [100, 999], "image_size": [512, 512],
                                     "font_size": 24, "bar_height": "floor(.28 * value)"},
                "verifier_probes_sha256": file_digest(destination / "verifier_probes.json")}
    manifest["data_signature"] = digest(manifest)
    atomic_json(destination / "manifest.json", manifest, immutable=True)
    verify(destination)
    return manifest


def verify(destination: Path) -> dict:
    manifest = read_json(destination / "manifest.json")
    signature = manifest["data_signature"]
    if digest({k: v for k, v in manifest.items() if k != "data_signature"}) != signature:
        raise ValueError("Dataset manifest signature mismatch")
    seen_scenes, seen_cases = set(), set()
    for name, split in manifest["splits"].items():
        path = destination / split["file"]
        if file_digest(path) != split["sha256"]:
            raise ValueError("Input parquet hash mismatch")
        rows = pq.read_table(path).to_pylist()
        if len(rows) != split["rows"]:
            raise ValueError("Split row count mismatch")
        for row in rows:
            info = row["extra_info"]
            if info["split"] != name or info["scene_id"] in seen_scenes or info["case_id"] in seen_cases:
                raise ValueError("Scene/case overlap or split mismatch")
            if row["reward_model"]["ground_truth"] != info["answer"]:
                raise ValueError("Checker metadata mismatch")
            import hashlib
            if hashlib.sha256(row["images"][0]["bytes"]).hexdigest() != info["image_sha256"]:
                raise ValueError("Image identity mismatch")
            seen_scenes.add(info["scene_id"])
            seen_cases.add(info["case_id"])
    if file_digest(destination / "verifier_probes.json") != manifest["verifier_probes_sha256"]:
        raise ValueError("Verifier probe hash mismatch")
    for split, probe in manifest["verifier_splits"].items():
        path = destination / probe["file"]
        if file_digest(path) != probe["sha256"] or pq.read_metadata(path).num_rows != probe["rows"]:
            raise ValueError("Verifier diagnostic parquet identity mismatch")
        base_ids = {r["case_id"] for r in manifest["samples"] if r["split"] == split}
        for row in pq.read_table(path).to_pylist():
            if row["extra_info"]["base_case_id"] not in base_ids:
                raise ValueError("Verifier probe crosses its permitted split")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=VERSION / "inputs/visual_baseline_08")
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()
    manifest = prepare(args.output.resolve(), args.seed)
    print({"data_signature": manifest["data_signature"], "splits": manifest["splits"]})
