import requests, time, sys

urls = [
    "https://download.pytorch.org/whl/nightly/rocm7.14/rocm-sdk-core/rocm_sdk_core-7.14.0-py3-none-win_amd64.whl",
    "https://download.pytorch.org/whl/nightly/rocm7.14/rocm-sdk-libraries/rocm_sdk_libraries-7.14.0-py3-none-win_amd64.whl",
    "https://download.pytorch.org/whl/nightly/rocm7.14/rocm-sdk-device-gfx1030/rocm_sdk_device_gfx1030-7.14.0-py3-none-win_amd64.whl",
]
s = requests.Session()
s.headers["User-Agent"] = "python-requests/2 (pip)"
for u in urls:
    t0 = time.perf_counter()
    r = s.get(u, headers={"Range": "bytes=0-1048575"}, stream=True, timeout=(15, 20))
    got = 0
    for chunk in r.iter_content(1 << 20):
        got += len(chunk)
    dt = time.perf_counter() - t0
    cr = r.headers.get("Content-Range", "?")
    print(
        f"{u.rsplit('/', 1)[-1]}: status={r.status_code} range={cr} "
        f"got={got/2**20:.2f}MB in {dt:.2f}s -> {got/dt/2**20:.2f} MB/s",
        flush=True,
    )
