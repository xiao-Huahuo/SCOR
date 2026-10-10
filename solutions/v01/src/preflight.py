"""Read-only GPU, disk and dependency checks before a pilot."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata as metadata
import json
import platform
import shutil
import subprocess
from pathlib import Path

from pilot_utils import atomic_json


def inspect(workspace: Path, gpu_ids: list[int], minimum_disk_gib: int = 55) -> dict:
    if len(gpu_ids) > 4 or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("Select at most four distinct GPUs")
    result = {"system": platform.platform(), "python": platform.python_version(),
              "workspace": str(workspace), "selected_gpus": gpu_ids,
              "free_disk_gib": round(shutil.disk_usage(workspace).free / 2**30, 2),
              "dependencies": {}, "gpu_rows": [], "problems": []}
    for name in ["torch", "vllm", "transformers", "flash-attn", "ray", "tensordict", "torchdata"]:
        try:
            result["dependencies"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result["dependencies"][name] = None
    command = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
               "--format=csv,noheader,nounits"]
    try:
        output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
        for row in csv.reader(output.splitlines()):
            index, name, total, free, utilization = [part.strip() for part in row]
            result["gpu_rows"].append({"index": int(index), "name": name,
                                       "total_mib": int(total), "free_mib": int(free),
                                       "utilization_percent": int(utilization)})
        for index in gpu_ids:
            row = next((row for row in result["gpu_rows"] if row["index"] == index), None)
            if row is None or row["free_mib"] < 22000 or row["utilization_percent"] > 5:
                result["problems"].append(f"GPU {index} is unavailable or busy")
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        result["problems"].append(f"GPU query failed: {type(error).__name__}")
    if platform.system() != "Linux":
        result["problems"].append("Native training requires Linux/CUDA")
    for name, version in result["dependencies"].items():
        if version is None:
            result["problems"].append(f"Missing dependency: {name}")
    # One 3B fp32 model + Adam checkpoint is about 45 GB. Keep room for transient files.
    if result["free_disk_gib"] < minimum_disk_gib:
        result["problems"].append(f"Less than {minimum_disk_gib} GiB free for the requested pilot")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect(args.workspace.resolve(), [int(i) for i in args.gpus.split(",")])
    if args.output:
        atomic_json(args.output, report, immutable=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(1 if report["problems"] else 0)
