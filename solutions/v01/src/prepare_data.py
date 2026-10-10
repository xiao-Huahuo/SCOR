"""Generate a small visual arithmetic dataset in the native verl schema."""
from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import pyarrow as pa
import pyarrow.parquet as pq

from pilot_utils import VERSION, atomic_json, digest, file_digest


def make_row(index: int, split: str, rng: random.Random) -> tuple[dict, dict]:
    # Sampling without replacement below keeps train and evaluation scenes disjoint.
    values = rng.sample(range(2, 25), 3)
    colors = ["red", "blue", "green"]
    family = index % 3
    image = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=24)
    draw.text((28, 20), "VALUES", fill="black", font=font)
    for i, (value, color) in enumerate(zip(values, colors)):
        x = 65 + i * 145
        height = value * 12
        draw.rectangle((x, 410 - height, x + 80, 410), fill=color)
        draw.text((x + 15, 420), color, fill="black", font=font)
        draw.text((x + 25, 370 - height), str(value), fill="black", font=font)
    if family == 0:
        question = "What is the sum of the values of the red and blue bars?"
        answer = values[0] + values[1]
    elif family == 1:
        question = "What is the product of the values of the blue and green bars?"
        answer = values[1] * values[2]
    else:
        question = "What is the largest bar value minus the smallest bar value?"
        answer = max(values) - min(values)
    prompt = (
        "<image>\n" + question + " Read the numeric labels from the image. "
        "Use a fenced Python code block to calculate and print the result. "
        "When giving the final answer, write it as \\boxed{number}."
    )
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    sample_id = digest({"values": values, "family": family})
    row = {
        "data_source": "scor_visual_arithmetic_pilot",
        "prompt": [{"role": "user", "content": prompt}],
        "images": [{"bytes": buffer.getvalue(), "path": None}],
        "ability": "visual_arithmetic",
        "reward_model": {"style": "rule", "ground_truth": str(answer)},
        "extra_info": {"index": index, "split": split, "sample_id": sample_id},
    }
    receipt = {"index": index, "split": split, "sample_id": sample_id,
               "scene_id": digest(values),
               "values": values, "family": family, "answer": str(answer),
               "image_sha256": hashlib.sha256(buffer.getvalue()).hexdigest()}
    return row, receipt


def prepare(destination: Path, seed: int = 20261010,
            train_count: int = 16, val_count: int = 4) -> dict:
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Use a new dataset directory; existing inputs are immutable")
    destination.mkdir(parents=True, exist_ok=True)
    rng, seen, splits, receipts = random.Random(seed), set(), {}, []
    for split, count in [("train", train_count), ("val", val_count)]:
        rows = []
        while len(rows) < count:
            row, receipt = make_row(len(receipts), split, rng)
            if receipt["scene_id"] in seen:
                continue
            seen.add(receipt["scene_id"])
            rows.append(row)
            receipts.append(receipt)
        path = destination / f"{split}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
        splits[split] = {"rows": count, "file": path.name,
                         "sha256": file_digest(path)}
    manifest = {"kind": "synthetic engineering pilot", "seed": seed,
                "not_a_paper_benchmark": True, "splits": splits,
                "samples": receipts, "data_signature": digest(receipts)}
    atomic_json(destination / "manifest.json", manifest, immutable=True)
    preview = pq.read_table(destination / "train.parquet").to_pylist()[0]["images"][0]["bytes"]
    (destination / "preview.png").write_bytes(preview)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=VERSION / "inputs" / "visual_pilot_v2")
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    result = prepare(args.output.resolve(), args.seed)
    print({"data_signature": result["data_signature"], "splits": result["splits"]})
