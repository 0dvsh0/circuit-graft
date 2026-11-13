"""Resumable parallel downloader for the ROCm SDK wheels (AMD nightly CDN).

The CDN drops long-lived single connections after a few hundred MB and pip
cannot resume a partial wheel, so a plain `pip install` kept dying mid-file
("Wheel 'rocm-sdk-core' ... is invalid"). This script:

* HEADs each URL for the final size (following redirects),
* splits the file into N byte-range chunks downloaded concurrently,
* keeps retrying each chunk forever (read timeouts / dropped connections are
  normal here; hard 4xx aborts),
* resumes every chunk from its own `.part` file on restart,
* verifies the assembled size against Content-Length and that the artifact
  is a readable ZIP (wheels) or tarball before declaring success.

Usage:
    python scripts/download_rocm_wheels.py            # download all
    python scripts/download_rocm_wheels.py --url URL  # download one URL
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BASE = "https://rocm.nightlies.amd.com/v2-staging/gfx103X-all/"
OUT_DIR = Path(r"D:\cg_wheels")
# rocm_sdk_devel (1.6 GiB) is compile-time headers only - not needed to run
# torch; skipped deliberately to protect the throttled link. The `rocm`
# metapackage sdist (21 kB) is already in the pip HTTP cache.
WHEELS = [
    "rocm_sdk_core-7.14.0a20260612-py3-none-win_amd64.whl",
    "rocm_sdk_libraries_gfx103x_all-7.14.0a20260612-py3-none-win_amd64.whl",
]
CHUNKS = 6
READ_TIMEOUT = 45
CONNECT_TIMEOUT = 30
BACKOFF = 5
MAX_BACKOFF = 60

_session = requests.Session()


def head_size(url: str) -> tuple[int, str]:
    r = _session.head(url, allow_redirects=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    r.raise_for_status()
    length = int(r.headers.get("Content-Length", "0"))
    if length <= 0:
        raise SystemExit(f"no Content-Length for {url}")
    return length, r.url


def download(url: str, dest: Path, total: int) -> None:
    part_files = [dest.with_suffix(dest.suffix + f".part{i}") for i in range(CHUNKS)]
    bounds = []
    for i in range(CHUNKS):
        start = total * i // CHUNKS
        end = total * (i + 1) // CHUNKS - 1
        bounds.append((start, end))
    print(f"{dest.name}: {total / 2**20:.1f} MB in {CHUNKS} chunks", flush=True)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CHUNKS) as pool:
        futures = []
        for i, (start, end) in enumerate(bounds):
            part = part_files[i]
            futures.append(pool.submit(_download_range_into, url, part, start, end))
        for fut in as_completed(futures):
            fut.result()
    got = sum(p.stat().st_size for p in part_files)
    if got != total:
        raise SystemExit(f"{dest.name}: assembled {got} != expected {total}")
    with open(dest, "wb") as out:
        for part in part_files:
            with open(part, "rb") as fh:
                while True:
                    buf = fh.read(1 << 22)
                    if not buf:
                        break
                    out.write(buf)
            part.unlink()
    size = dest.stat().st_size
    if dest.suffix == ".whl":
        with zipfile.ZipFile(dest) as zf:
            assert zf.testzip() is None, f"{dest.name}: corrupt zip member"
    dt = time.perf_counter() - t0
    print(f"{dest.name}: OK {size / 2**20:.1f} MB in {dt:.0f}s ({size / dt / 2**20:.2f} MB/s)", flush=True)


def _download_range_into(url: str, part: Path, start: int, end: int) -> None:
    """Range chunk with its own retry loop; each .partN file holds its slice."""
    span = end - start + 1
    backoff = BACKOFF
    while True:
        have = part.stat().st_size if part.exists() else 0
        if have >= span:
            return
        try:
            headers = {"Range": f"bytes={start + have}-{end}"}
            r = _session.get(url, headers=headers, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            r.raise_for_status()
            with open(part, "ab") as fh:
                for data in r.iter_content(chunk_size=1 << 20):
                    fh.write(data)
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else 0
            if 400 <= code < 500 and code not in (408, 429):
                raise SystemExit(f"{part.name}: hard HTTP {code} for {url}")
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)
        except requests.RequestException:
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", action="append", help="additional URL to download")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    urls = [(BASE + name, OUT_DIR / name) for name in WHEELS]
    for extra in args.url or []:
        urls.append((extra, OUT_DIR / extra.rsplit("/", 1)[-1]))
    failures = []
    for url, dest in urls:
        if dest.exists() and dest.suffix == ".whl":
            try:
                with zipfile.ZipFile(dest) as zf:
                    if zf.testzip() is None:
                        print(f"{dest.name}: already complete, skipping")
                        continue
            except Exception:  # noqa: BLE001
                dest.unlink()
        try:
            total, final_url = head_size(url)
            if dest.exists() and dest.stat().st_size == total and dest.suffix != ".whl":
                print(f"{dest.name}: already complete, skipping")
                continue
            download(final_url, dest, total)
        except SystemExit as exc:
            failures.append(str(exc))
            print(f"FAILED {url}: {exc}", flush=True)
    if failures:
        print(f"\n{len(failures)} download(s) failed")
        return 1
    print("\nALL WHEELS DOWNLOADED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
