import re

import requests

for path in [
    "rocm/whl-multi-arch/rocm-sdk-core/",
    "rocm/whl-multi-arch/rocm-sdk-libraries/",
    "rocm/whl-multi-arch/rocm-sdk-device-gfx1030/",
]:
    r = requests.get("https://repo.amd.com/" + path, timeout=30)
    print("===", path, r.status_code)
    if r.status_code != 200:
        print(r.text[:300])
        continue
    wins = [m.group(1).split("/")[-1] for m in re.finditer(r'href="([^"]+)"', r.text) if "win_amd64" in m.group(1)]
    rel = [w for w in wins if re.search(r"7\.1[34]", w)]
    for w in rel[-8:]:
        print("  ", w)
