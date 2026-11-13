import time

import requests

base = "https://repo.amd.com/rocm/whl-multi-arch/"
files = [
    "rocm_sdk_core-7.14.0-py3-none-win_amd64.whl",
    "rocm_sdk_libraries-7.14.0-py3-none-win_amd64.whl",
    "rocm_sdk_device_gfx1030-7.14.0-py3-none-win_amd64.whl",
]
s = requests.Session()
s.headers["User-Agent"] = "python-requests/2 (pip)"
for name in files:
    u = base + name
    t0 = time.perf_counter()
    r = s.get(u, headers={"Range": "bytes=0-2097151"}, stream=True, timeout=(15, 25))
    got = 0
    for chunk in r.iter_content(1 << 20):
        got += len(chunk)
    dt = time.perf_counter() - t0
    total = r.headers.get("Content-Range", "?")
    accept = r.headers.get("Accept-Ranges", "?")
    print(
        f"{name}: status={r.status_code} total-range={total} accept-ranges={accept} "
        f"got={got/2**20:.2f}MB in {dt:.2f}s -> {got/dt/2**20:.2f} MB/s",
        flush=True,
    )
