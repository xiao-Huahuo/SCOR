"""Version-local identities and immutable native baseline records."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

from pilot_utils import digest, file_digest


def stream_seed(seed: int, phase: str, step: int, case_id: str,
                slot: int, role: str, inner_step: int) -> int:
    # Branch/run names never change matched training streams.
    return int(digest([seed, phase, step, case_id, slot, role, inner_step])[:8], 16)


def trajectory_id(step: int, case_id: str, slot: int) -> str:
    return f"{step}:{case_id}:{slot}"


def tensor_save(value, path: Path) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        fsync_directory(path.parent)
    finally:
        os.unlink(temporary)


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def value_hash(value) -> str:
    """Hash actual tensors, image pixels, arrays and nested input structure."""
    import numpy as np
    if isinstance(value, dict):
        return digest({str(k): value_hash(v) for k, v in sorted(value.items())})
    if isinstance(value, (list, tuple)):
        return digest([value_hash(v) for v in value])
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return digest({"shape": value.shape, "values": value_hash(value.tolist())})
        return digest({"dtype": str(value.dtype), "shape": value.shape,
                       "bytes": hashlib.sha256(value.tobytes()).hexdigest()})
    if isinstance(value, np.generic):
        return value_hash(value.item())
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    if hasattr(value, "detach"):
        import torch
        tensor = value.detach().cpu().contiguous()
        return digest({"dtype": str(tensor.dtype), "shape": list(tensor.shape),
                       "bytes": hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()})
    if hasattr(value, "tobytes") and hasattr(value, "mode"):
        return digest({"mode": value.mode, "size": value.size,
                       "pixels": hashlib.sha256(value.tobytes()).hexdigest()})
    if isinstance(value, (str, int, float, bool)) or value is None:
        return digest(value)
    raise TypeError(f"Unsupported evidence value: {type(value).__name__}")


def prompt_hash(data) -> str:
    return value_hash({"tensors": dict(data.batch.items()), "inputs": data.non_tensor_batch})


def model_identity(model: Path) -> dict:
    required = {"config.json", "tokenizer_config.json", "preprocessor_config.json"}
    if not model.is_dir() or not all((model / p).is_file() for p in required):
        raise ValueError("Pass a complete local multimodal model snapshot")
    files = {str(p.relative_to(model)): {"bytes": p.stat().st_size, "sha256": file_digest(p)}
             for p in sorted(model.rglob("*")) if p.is_file() and ".cache" not in p.relative_to(model).parts}
    shards = [name for name in files if name.endswith(".safetensors")]
    if not shards:
        raise ValueError("Model weights are missing")
    from pilot_utils import read_json
    index = model / "model.safetensors.index.json"
    if index.exists() and not set(read_json(index)["weight_map"].values()).issubset(shards):
        raise ValueError("Model snapshot has missing weight shards")
    return {"files": files, "signature": digest(files),
            "weight_bytes": sum(files[name]["bytes"] for name in shards)}


def signatures(learning: dict, protocol: dict, parent: dict | None) -> dict:
    return {"learning": digest(learning), "protocol": digest(protocol),
            "parent": digest(parent), "run": digest([learning, protocol, parent])}
