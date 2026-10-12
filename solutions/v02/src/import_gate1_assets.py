"""Verify and physically materialize immutable v01 inputs for Gate1."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil

from checkpoint_store import validate
from pilot_utils import ROOT, VERSION, atomic_json, digest, file_digest, read_json
from records import prompt_hash
from gate1_contract import historical_output_matches


def link_copy(source: Path, target: Path, hashes: dict, destination: Path):
    if source.is_symlink() or not source.is_file():
        raise ValueError("Only actual owned files can be imported")
    target.parent.mkdir(parents=True, exist_ok=True)
    sha = file_digest(source)
    try:
        os.link(source, target)
    except OSError:
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
            outgoing.flush(); os.fsync(outgoing.fileno())
    if file_digest(target) != sha:
        raise ValueError("Materialized source differs")
    hashes[str(target.relative_to(destination))] = {"source": str(source.resolve()),
        "bytes": target.stat().st_size, "sha256": sha, "hardlinked": os.path.samefile(source, target)}


def import_assets(source_version: Path, destination: Path) -> dict:
    import torch
    from prepare_baseline_data import verify
    if not source_version.resolve().is_relative_to(ROOT.resolve()) or not destination.resolve().is_relative_to((VERSION/"inputs").resolve()):
        raise ValueError("Use owned source assets and version-local inputs")
    destination.mkdir(parents=True, exist_ok=False)
    hashes, histories, states = {}, {}, {}
    canonical = None
    try:
        source_data = source_version/"inputs/visual_baseline_08"
        data_identity = verify(source_data)
        for path in sorted(source_data.rglob("*")):
            if path.is_file():
                link_copy(path, destination/"visual_baseline_08"/path.relative_to(source_data), hashes, destination)
        if verify(destination/"visual_baseline_08") != data_identity:
            raise ValueError("Imported dataset identity differs")
        for seed in [20261011, 20261012, 20261013]:
            source = source_version/f"outputs/baseline_formal_20261011_seed_{seed}"
            manifest = read_json(source/"manifest.json")
            learning = manifest["signatures"]["learning"]
            if digest(manifest["learning"]) != learning or manifest["seed"] != seed:
                raise ValueError("History learning/seed identity differs")
            if read_json(source/"status.json") != {"status": "completed", "returncode": 0}:
                raise ValueError("Source history did not complete normally")
            if canonical is None:
                canonical = manifest["learning"]
            template = dict(canonical); template["seed"] = seed
            if manifest["learning"] != template or manifest["data_signature"] != data_identity["data_signature"]:
                raise ValueError("Historical source conditions differ across seeds")
            for name, sha in manifest["learning"]["source_sha256"].items():
                if file_digest(source/"raw/source"/name) != sha or file_digest(source_version/"src"/name) != sha:
                    raise ValueError("Canonical frozen engine differs: " + name)
            target = destination/f"histories/seed_{seed}"
            link_copy(source/"manifest.json", target/"manifest.json", hashes, destination)
            link_copy(source/"status.json", target/"status.json", hashes, destination)
            for step in range(1,33):
                payload = torch.load(source/f"raw/generation_step_{step:06d}.pt", map_location="cpu", weights_only=False)
                # The original prompt itself is not saved; load_generation needs the live dataloader
                # witness. Here verify output against the independently committed training group.
                group = torch.load(source/f"raw/training_group_step_{step:06d}.pt", map_location="cpu", weights_only=False)
                completed = read_json(source/f"raw/completed_step_{step:06d}.json")
                group_receipt = read_json(source/f"raw/group_step_{step:06d}.json")
                if (payload["learning_signature"] != learning or payload["step"] != step or
                        completed["learning_signature"] != learning or
                        completed["group_sha256"] != file_digest(source/f"raw/group_step_{step:06d}.json") or
                        prompt_hash(group) != group_receipt["payload_hash"]):
                    raise ValueError("Incomplete/corrupt historical unit")
                historical_output_matches(payload["output"], group)
                for stem, extension in [("generation", "pt"), ("training_group", "pt"),
                                        ("completed", "json"), ("group", "json"), ("reward", "json"), ("metrics", "json")]:
                    name=f"raw/{stem}_step_{step:06d}.{extension}"
                    link_copy(source/name, target/name, hashes, destination)
            histories[str(seed)] = {"path": str(target.resolve()), "source": str(source.resolve()),
                                    "learning_signature": learning, "units": 32}
        for label, run_name, step in [
            ("seed_20261011_S7", "baseline_formal_20261011_seed_20261011", 7),
            ("seed_20261011_clean_S8", "baseline_formal_20261011_seed_20261011", 8),
            ("seed_20261011_corrupt_S8", "baseline_formal_20261011_E_delta1", 8)]:
            source_run=source_version/"outputs"/run_name
            source=source_run/f"checkpoints/global_step_{step}"
            cert=validate(source)
            target=destination/"states"/label/source.name
            for path in sorted(source.rglob("*")):
                if path.is_file(): link_copy(path,target/path.relative_to(source),hashes,destination)
            link_copy(source_run/"manifest.json",target.parent/"source_manifest.json",hashes,destination)
            if validate(target)!=cert:
                raise ValueError("Imported complete state differs")
            states[label] = {"path": str(target.resolve()), "state_signature": cert["state_signature"],
                             "learning_signature": cert["learning_signature"], "step": step}
        receipt={"status":"verified", "created_utc":datetime.now(timezone.utc).isoformat(),
                 "source_version":str(source_version.resolve()), "destination":str(destination.resolve()),
                 "files":hashes,"histories":histories,"states":states,
                 "canonical_learning":canonical,"canonical_source_sha256":canonical["source_sha256"],
                 "dataset_signature":data_identity["data_signature"],
                 "missing_prefix_seeds":[20261012,20261013],"entry_sha256":file_digest(Path(__file__))}
        receipt["manifest_signature"]=digest(receipt)
        atomic_json(destination/"import_manifest.json",receipt,immutable=True)
        return receipt
    except BaseException as error:
        atomic_json(destination/"import_failed.json", {"error_type":type(error).__name__,"error":str(error)},immutable=True)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-version",type=Path,default=VERSION.parent/"v01")
    parser.add_argument("--output",type=Path,default=VERSION/"inputs/import_v01_20261012")
    args=parser.parse_args()
    print(import_assets(args.source_version,args.output)["manifest_signature"])


if __name__=="__main__": main()
