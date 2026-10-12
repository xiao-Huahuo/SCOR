"""Stream a complete server snapshot to a verified local lossless archive."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import time

from pilot_utils import ROOT, VERSION, atomic_json, file_digest, read_json
from checkpoint_store import validate


def archive(run_id: str, step: int, *, retire_server: bool = False) -> dict:
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id) or step < 0:
        raise ValueError("Invalid own snapshot identity")
    server = read_json(ROOT / "CURRENT_STATE.json")["server"]
    relative = f"solutions/v01/outputs/{run_id}/checkpoints/global_step_{step}"
    root, alias = server["workspace"], server["alias"]
    prefix = ["ssh", "-F", str(ROOT / ".runtime/ssh/config"), "-o", "BatchMode=yes", alias]
    remote = shlex.quote(root + "/.runtime/gpu-venv/bin/python") + " -c " + shlex.quote(
        "import sys,json,pathlib;sys.path.insert(0," + repr(root + "/solutions/v01/src") + ");"
        "from checkpoint_store import validate;print(json.dumps(validate(pathlib.Path(" + repr(root + "/" + relative) + "))))")
    certificate = json.loads(subprocess.check_output([*prefix, remote], text=True))
    local = VERSION / "outputs" / run_id / "archives"
    local.mkdir(parents=True, exist_ok=True)
    output = local / f"global_step_{step}.tar.gz"
    if output.exists():
        raise FileExistsError("Archive already exists; reuse its verified receipt")
    partial = output.with_suffix(output.suffix + ".downloading")
    begin = time.monotonic()
    command = "tar -czf - -C " + shlex.quote(root + "/" + relative) + " ."
    with partial.open("xb") as stream:
        subprocess.run([*prefix, command], stdout=stream, check=True)
    extraction = local / f"verify_{step}"
    extraction.mkdir(exist_ok=False)
    with tarfile.open(partial, "r:gz") as bundle:
        bundle.extractall(extraction, filter="data")
    materialized = extraction / f"global_step_{step}"
    # The archive contains the snapshot root; move members into its canonical name.
    materialized.mkdir()
    for path in list(extraction.iterdir()):
        if path != materialized:
            path.rename(materialized / path.name)
    actual = validate(materialized, certificate["learning_signature"])
    if actual != certificate:
        raise ValueError("Archived complete state differs from server certificate")
    partial.replace(output)
    receipt = {"run_id": run_id, "step": step, "source": root + "/" + relative,
               "state_signature": certificate["state_signature"], "archive_sha256": file_digest(output),
               "archive_bytes": output.stat().st_size, "uncompressed_bytes": sum(x["bytes"] for x in certificate["files"].values()),
               "all_members_sha256_verified": True, "wall_seconds": time.monotonic() - begin,
               "server_copy_retired": False}
    atomic_json(local / f"global_step_{step}_archive.json", receipt, immutable=True)
    atomic_json(local / f"global_step_{step}_certificate.json", certificate, immutable=True)
    atomic_json(local / f"global_step_{step}_worker_state.json",
                read_json(materialized / "baseline_worker_state.json"), immutable=True)
    shutil.rmtree(extraction)  # Own verification temporary; verified lossless archive remains.
    if retire_server:
        source = root + "/" + relative
        code = (
            "import sys,json,pathlib,shutil;sys.path.insert(0," + repr(root + "/solutions/v01/src") + ");"
            "from checkpoint_store import validate;root=pathlib.Path(" + repr(root) + ");"
            "p=pathlib.Path(" + repr(source) + ");c=validate(p);"
            "assert p.resolve().is_relative_to(root/'solutions/v01/outputs');"
            "assert c['state_signature']==" + repr(certificate["state_signature"]) + ";"
            "status=json.loads((root/'solutions/v01/outputs'/" + repr(run_id) + "/'status.json').read_text());"
            "assert status['status'] in ['completed','intentionally_stopped','failed'];"
            "from pilot_utils import atomic_json;"
            "atomic_json(root/'solutions/v01/outputs'/" + repr(run_id) + "/'raw'/" + repr(f"archived_step_{step:06d}.json") + ","
            + repr({"state_signature": certificate["state_signature"], "verified_local_archive": str(output),
                    "archive_sha256": receipt["archive_sha256"], "all_members_verified": True}) + ",immutable=True);"
            "shutil.rmtree(p);print('Verified local backup retained; server replica retired')")
        subprocess.run([*prefix, shlex.quote(root + "/.runtime/gpu-venv/bin/python") + " -c " + shlex.quote(code)], check=True)
        atomic_json(local / f"global_step_{step}_server_retirement.json",
                    {"server_copy_retired": True, "archive_sha256": receipt["archive_sha256"]}, immutable=True)
        receipt["server_copy_retired"] = True
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--retire-server", action="store_true")
    args = parser.parse_args()
    archive(args.run_id, args.step, retire_server=args.retire_server)
