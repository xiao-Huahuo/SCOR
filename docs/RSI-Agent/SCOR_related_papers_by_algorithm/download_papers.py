from pathlib import Path
import csv
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "papers_manifest.tsv"
CACHE = ROOT / ".pdf_cache"
CACHE.mkdir(exist_ok=True)

rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8"), delimiter="\t"))

def valid_pdf(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size >= 1000 and path.open("rb").read(5) == b"%PDF-"
    except Exception:
        return False

def download_with_curl(url: str, out: Path) -> None:
    part = out.with_suffix(out.suffix + ".part")
    part.unlink(missing_ok=True)
    cmd = [
        "curl", "-L", "--fail", "--show-error", "--progress-bar",
        "--retry", "2", "--retry-delay", "2", "--retry-all-errors",
        "--connect-timeout", "10", "--max-time", "180",
        "-A", "Mozilla/5.0",
        "-o", str(part), url,
    ]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"curl exited with code {proc.returncode}")
    if not valid_pdf(part):
        part.unlink(missing_ok=True)
        raise RuntimeError("downloaded response is not a valid PDF")
    part.replace(out)

if shutil.which("curl") is None:
    print("[error] curl was not found. macOS normally includes curl.")
    sys.exit(2)

ok = fail = 0
unique_downloads = {}
for i, row in enumerate(rows, 1):
    dest = ROOT / row["destination"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = row["pdf_url"].strip()
    cache_name = f'{row["global_number"]}_{row["short_name"]}.pdf'
    cached = CACHE / cache_name

    print(f"\n[{i}/{len(rows)}] {row['short_name']}", flush=True)
    print(f"  -> {dest.relative_to(ROOT)}", flush=True)

    if valid_pdf(dest):
        print("  [skip] already downloaded", flush=True)
        ok += 1
        continue

    # Reuse the same PDF when one paper appears under multiple SCOR sub-algorithms.
    if valid_pdf(cached):
        shutil.copy2(cached, dest)
        print("  [copy] reused cached PDF", flush=True)
        ok += 1
        continue

    print(f"  [download] {url}", flush=True)
    try:
        download_with_curl(url, cached)
        shutil.copy2(cached, dest)
        size_mb = cached.stat().st_size / (1024 * 1024)
        print(f"  [ok] {size_mb:.1f} MB", flush=True)
        ok += 1
    except KeyboardInterrupt:
        print("\n[stopped] interrupted by user", flush=True)
        sys.exit(130)
    except Exception as e:
        print(f"  [failed] {e}", flush=True)
        fail += 1

print(f"\ndone: {ok} ok, {fail} failed", flush=True)
print("Failed papers can be retried by running this script again; completed PDFs are skipped.", flush=True)
sys.exit(1 if fail else 0)
