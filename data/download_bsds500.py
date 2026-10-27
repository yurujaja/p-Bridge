"""One-shot downloader for BSDS500 from Berkeley (BSR_bsds500.tgz, ~70 MB).

Extracts train + val + test images into `data/bsds500/{train,val,test}/*.jpg`.
Idempotent — skips download/extract if the target dirs already have images.
"""

from __future__ import annotations
import os
import ssl
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"

ROOT = Path(__file__).resolve().parents[1] / "data"
DL_DIR = ROOT / "downloads"
OUT = ROOT / "bsds500"
SPLITS = ("train", "val", "test")


def _extract(tgz_path: Path, out_root: Path):
    out_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz_path, "r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            # Files look like BSR/BSDS500/data/images/train/12345.jpg
            parts = m.name.split("/")
            try:
                idx_images = parts.index("images")
                split = parts[idx_images + 1]
                base = parts[-1]
            except (ValueError, IndexError):
                continue
            if split not in SPLITS or not base.lower().endswith(".jpg"):
                continue
            dst_dir = out_root / split
            dst_dir.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(m) as f:
                (dst_dir / base).write_bytes(f.read())


def ensure_bsds500(force: bool = False) -> Path:
    """Return the BSDS500 root; download+extract if not present."""
    if not force and all((OUT / s).exists() and any((OUT / s).iterdir()) for s in SPLITS):
        return OUT
    DL_DIR.mkdir(parents=True, exist_ok=True)
    tgz = DL_DIR / "BSR_bsds500.tgz"
    if force or not tgz.exists() or tgz.stat().st_size < 10_000_000:
        print(f"downloading {URL} -> {tgz}", flush=True)
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(URL, context=ctx, timeout=60) as r, open(tgz, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            got = 0
            chunk = 1024 * 512
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                got += len(buf)
                if total:
                    pct = 100.0 * got / total
                    if int(got / chunk) % 20 == 0:
                        print(f"  {got/1e6:6.1f} / {total/1e6:.1f} MB  ({pct:5.1f}%)", flush=True)
        print("download done", flush=True)
    print(f"extracting -> {OUT}", flush=True)
    _extract(tgz, OUT)
    counts = {s: len(list((OUT / s).glob("*.jpg"))) for s in SPLITS}
    print(f"extracted: {counts}", flush=True)
    return OUT


if __name__ == "__main__":
    ensure_bsds500(force="--force" in sys.argv)
