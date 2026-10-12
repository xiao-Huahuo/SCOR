"""Download the exact public Qwen weights and verify primary HF file hashes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import urllib.parse
import urllib.request

from pilot_utils import atomic_json, file_digest, read_json


def download(path: Path, url: str, expected: dict) -> None:
    if path.exists() and path.stat().st_size == expected["bytes"] and file_digest(path) == expected["sha256"]:
        return
    partial = path.with_suffix(path.suffix + ".downloading")
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == expected["bytes"]:
            break
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45) as response:
                append = offset > 0 and response.status == 206
                with partial.open("ab" if append else "wb") as stream:
                    while block := response.read(4 * 1024 * 1024):
                        stream.write(block)
            if partial.stat().st_size == expected["bytes"]:
                break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(attempt + 1)
    if partial.stat().st_size != expected["bytes"] or file_digest(partial) != expected["sha256"]:
        raise ValueError("Model download size/hash differs from the primary source")
    partial.replace(path)


def prepare(destination: Path, metadata: dict) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    weights = {name: info for name, info in metadata["files"].items() if name.endswith(".safetensors")}
    def fetch(item):
        name, info = item
        url = "https://modelscope.cn/api/v1/models/Qwen/Qwen2-VL-2B-Instruct/repo?" + urllib.parse.urlencode(
            {"Revision": "master", "FilePath": name})
        download(destination / name, url, info)
        print(f"Verified model weight: {name}", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fetch, weights.items()))
    for name, info in metadata["files"].items():
        path = destination / name
        if not path.is_file() or path.stat().st_size != info["bytes"] or file_digest(path) != info["sha256"]:
            raise ValueError("Model snapshot file differs from the primary source: " + name)
    receipt = {"repository": metadata["repository"], "revision": metadata["revision"],
               "weight_transport": "official Qwen ModelScope repository; exact HF LFS SHA-256 verified",
               "configuration_transport": "primary HF pinned revision", "files": metadata["files"]}
    marker = destination / "model_identity.json"
    if not marker.exists():
        atomic_json(marker, receipt, immutable=True)
    elif read_json(marker) != receipt:
        raise ValueError("Existing model identity differs")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.destination.resolve(), read_json(args.metadata))
    print("Complete model identity verified", flush=True)
