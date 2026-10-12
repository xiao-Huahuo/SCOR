"""Sample physical selected-device memory during an owned native run."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import time

from pilot_utils import ROOT, VERSION


def record(run: Path, pid: int, gpus: list[int]) -> None:
    run = run.resolve()
    if not run.is_relative_to((VERSION / "outputs").resolve()) or len(set(gpus)) != len(gpus) or not 1 <= len(gpus) <= 4:
        raise ValueError("Monitor only an owned version-local run on at most four GPUs")
    process = Path("/proc") / str(pid)
    if (process / "cwd").resolve() != ROOT or run.name.encode() not in (process / "cmdline").read_bytes():
        raise ValueError("PID does not belong to the selected own baseline")
    for _ in range(30):
        if (run / "raw").is_dir():
            break
        time.sleep(1)
    with (run / "raw/physical_gpu.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["utc", "gpu", "device_used_mib", "device_utilization_percent"])
        while process.exists():
            try:
                os.kill(pid, 0)
                state = (process / "status").read_text(encoding="utf-8")
                if any(line.startswith("State:") and line.split()[1] == "Z" for line in state.splitlines()):
                    break
            except (ProcessLookupError, FileNotFoundError):
                break
            output = subprocess.check_output(["nvidia-smi", "--id=" + ",".join(map(str, gpus)),
                "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
            stamp = datetime.now(timezone.utc).isoformat()
            writer.writerows([stamp, *(part.strip() for part in row)] for row in csv.reader(output.splitlines()))
            stream.flush()
            time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--gpus", default="4,5,6,7")
    args = parser.parse_args()
    record(args.run, args.pid, [int(x) for x in args.gpus.split(",")])
