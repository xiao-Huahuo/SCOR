"""Keep historical metadata honest while checking rebuilt native prefixes."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from gate1_evidence import evidence, member, physical
from gate1_state import certificate_metadata
from pilot_utils import ROOT, VERSION, atomic_json, digest, file_digest, read_json
from records import fsync_directory


def materialize(assets: Path, output: Path) -> dict:
    imported=read_json(assets)
    if (imported["status"]!="verified" or imported["manifest_signature"] != digest(
            {k:v for k,v in imported.items() if k!="manifest_signature"})):
        raise ValueError("Use accepted import and version-local reference metadata")
    output=physical(output,VERSION/"inputs")
    output.mkdir(parents=True,exist_ok=False)
    references={}
    for seed,history in imported["histories"].items():
        source=physical(Path(history["source"]))
        source_manifest=read_json(source/"manifest.json")
        if source_manifest["seed"]!=int(seed) or source_manifest["signatures"]["learning"]!=history["learning_signature"]:
            raise ValueError("Historical reference seed or learning identity differs")
        for step in [0,7,32]:
            snapshot=source/f"checkpoints/global_step_{step}"
            cert=certificate_metadata(snapshot)
            if (cert["state_signature"]!=digest({k:v for k,v in cert.items() if k!="state_signature"}) or
                    cert["step"]!=step or cert["learning_signature"]!=history["learning_signature"] or
                    cert["run_id"]!=source_manifest["run_id"]):
                raise ValueError("Historical certificate is corrupted")
            target=output/f"seed_{seed}"/f"global_step_{step}";target.mkdir(parents=True)
            members={}
            for name in ["baseline_worker_state.json","baseline_state.json","baseline_driver_rng.pt","data.pt"]:
                path=member(snapshot,name)
                if file_digest(path)!=cert["files"][name]["sha256"]:
                    raise ValueError("Historical reference metadata differs")
                os.link(path,target/name)
                members[name]={"sha256":file_digest(target/name),"bytes":path.stat().st_size}
            os.link(snapshot/"baseline_complete.json",target/"baseline_complete.json")
            fsync_directory(target)
            available=all(member(snapshot,name).is_file() and member(snapshot,name).stat().st_size==info["bytes"]
                          for name,info in cert["files"].items())
            references[f"{seed}:{step}"]={"path":str(target.resolve()),"source":str(snapshot.resolve()),
                "seed":int(seed),"run_id":source_manifest["run_id"],
                "state_signature":cert["state_signature"],"members":members,"metadata_only":True,
                "certificate_sha256":file_digest(target/"baseline_complete.json"),
                "source_all_required_files_present":available,
                "source_model_hashes_recomputed_by_this_metadata_import":False}
    receipt={"references":references,"import_manifest_sha256":file_digest(assets),
             "entry_sha256":file_digest(Path(__file__)),"scope":"historical certified metadata; not a restorable checkpoint"}
    receipt["manifest_signature"]=digest(receipt)
    atomic_json(output/"manifest.json",receipt,immutable=True)
    return receipt


def validate_reference(reference: Path) -> dict:
    reference=physical(reference,VERSION/"inputs")
    manifest=read_json(reference.parents[1]/"manifest.json")
    if manifest["manifest_signature"]!=digest({k:v for k,v in manifest.items() if k!="manifest_signature"}):
        raise ValueError("Historical reference manifest changed")
    matches=[entry for entry in manifest["references"].values() if physical(Path(entry["path"]))==reference]
    if len(matches)!=1:
        raise ValueError("Reference is not uniquely registered")
    entry=matches[0];certificate=certificate_metadata(reference)
    if (certificate["state_signature"]!=entry["state_signature"] or certificate["run_id"]!=entry["run_id"] or
            file_digest(reference/"baseline_complete.json")!=entry["certificate_sha256"] or
            entry["metadata_only"] is not True):
        raise ValueError("Historical reference certificate identity differs")
    for name,info in entry["members"].items():
        p=member(reference,name)
        if p.stat().st_size!=info["bytes"] or file_digest(p)!=info["sha256"]:
            raise ValueError("Historical reference metadata changed")
    return {"entry":entry,"certificate":certificate,"evidence":[evidence(reference.parents[1]/"manifest.json"),
            evidence(reference/"baseline_complete.json"),*[evidence(member(reference,name)) for name in entry["members"]]]}


def compare_rebuilt(actual: Path,reference: Path,*,allow_retired:bool=False) -> dict:
    import torch
    from gate1_state import state_fingerprint
    from records import value_hash
    verified=validate_reference(reference)
    fingerprint=state_fingerprint(actual,allow_retired=allow_retired)
    actual_manifest=read_json(actual.parent/"source_manifest.json" if (actual.parent/"source_manifest.json").exists()
                              else actual.parent.parent/"manifest.json")
    if actual_manifest["seed"]!=verified["entry"]["seed"]:
        raise ValueError("Rebuilt state belongs to a different historical training seed")
    checks={"step":fingerprint["step"]==verified["certificate"]["step"],
            "worker_model_buffers_Adam_groups_scheduler_rng":fingerprint["workers"]==read_json(reference/"baseline_worker_state.json"),
            "driver_rng":fingerprint["driver_rng"]==value_hash(torch.load(reference/"baseline_driver_rng.pt",map_location="cpu",weights_only=False)),
            "data_cursor":fingerprint["data_cursor"]==value_hash(torch.load(reference/"data.pt",map_location="cpu",weights_only=False))}
    state=read_json(reference/"baseline_state.json")
    checks["position"]=fingerprint["position"]=={key:state[key] for key in fingerprint["position"]}
    return {"passed":all(checks.values()),"checks":checks,
            "evidence":verified["evidence"],
            "reference_certificate_sha256":file_digest(reference/"baseline_complete.json"),
            "actual_certificate_sha256":file_digest(actual/"baseline_complete.json"),
            "scope":"rebuilt full state compared to historically certified complete fingerprints; no old weights availability claim"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=VERSION/"inputs/reference_metadata_20261012")
    args=parser.parse_args();print(materialize(args.assets,args.output)["manifest_signature"])


if __name__=="__main__":main()
