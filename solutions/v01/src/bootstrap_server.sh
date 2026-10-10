#!/usr/bin/env bash
set -euo pipefail
src="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd -- "$src/../../.." && pwd)"
test -f "$project/.scor_workspace"
mkdir -p "$project/.runtime/tmp" "$project/.runtime/wheels"
export TMPDIR="$project/.runtime/tmp"
export PIP_DISABLE_PIP_VERSION_CHECK=1
base_python="${SCOR_BASE_PYTHON:-python3}"
"$base_python" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use a Python 3.11 interpreter"'
if [[ ! -x "$project/.runtime/gpu-venv/bin/python" ]]; then
    "$base_python" -m venv "$project/.runtime/gpu-venv"
fi
python="$project/.runtime/gpu-venv/bin/python"
"$python" -m pip install --no-cache-dir 'pip==25.0.1' 'setuptools==75.8.0' 'wheel==0.45.1'
"$python" -m pip install --no-cache-dir 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
# xgrammar 0.1.16 is no longer on PyPI, but its exact official source tag exists.
# Build that version rather than changing the native vLLM / Transformers pair.
if ! "$python" -c 'from importlib.metadata import version; assert version("xgrammar") == "0.1.16"' 2>/dev/null; then
    "$python" -m pip install --no-cache-dir 'cmake>=3.26,<4' ninja 'scikit-build-core==0.10.7' 'pybind11==2.13.6'
    if [[ ! -d "$project/.runtime/upstream/xgrammar/.git" ]]; then
        git clone --depth 1 --branch v0.1.16 --recurse-submodules https://github.com/mlc-ai/xgrammar.git "$project/.runtime/upstream/xgrammar"
    fi
    git -C "$project/.runtime/upstream/xgrammar" describe --tags --exact-match
    git -C "$project/.runtime/upstream/xgrammar" submodule update --init --recursive
    export PATH="$project/.runtime/gpu-venv/bin:$PATH"
    CMAKE_BUILD_PARALLEL_LEVEL=4 "$python" -m pip wheel --no-cache-dir --no-deps --no-build-isolation "$project/.runtime/upstream/xgrammar" -w "$project/.runtime/wheels"
    "$python" -m pip install --no-cache-dir --no-deps "$project"/.runtime/wheels/xgrammar-0.1.16-*.whl
fi
"$python" -m pip install --no-cache-dir -r "$src/requirements-gpu.txt"
# Official FlashAttention release wheel: avoid building against a missing system nvcc.
"$python" - "$project/.runtime/wheels" <<'PY'
import json, pathlib, urllib.request, subprocess, sys, torch, zipfile
abi = str(bool(torch._C._GLIBCXX_USE_CXX11_ABI)).upper()
name = f'flash_attn-2.7.4.post1+cu12torch2.6cxx11abi{abi}-cp311-cp311-linux_x86_64.whl'
url = 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/' + name
path = pathlib.Path(sys.argv[1]) / name
if not path.exists() or not zipfile.is_zipfile(path):
    temporary = path.with_name(path.name + '.downloading')
    urllib.request.urlretrieve(url, temporary)
    if not zipfile.is_zipfile(temporary):
        raise RuntimeError('Incomplete FlashAttention wheel; not installed')
    temporary.replace(path)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', str(path)], check=True)
print(json.dumps({'flash_attention_wheel': name, 'torch': torch.__version__}))
PY
"$python" -m pip check
"$python" -c 'import torch, vllm, flash_attn; print({"torch":torch.__version__, "cuda_available":torch.cuda.is_available(), "vllm":vllm.__version__, "flash_attn":flash_attn.__version__})'
