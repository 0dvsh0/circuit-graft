# D1 smoke summary

- Generated: 2026-11-07T12:38:06.948993+00:00
- Mode: Tiny Aya
- Model: `CohereLabs/tiny-aya-global`
- G0 target: `CohereLabs/tiny-aya-global`
- Device: `cuda` (bfloat16)
- G0: **failed** - failed checks: circuitkit

| Check | Status | Wall clock (s) | Log |
|---|---|---|---|
| preflight | ok | 34.1 | `results/d1/smoke/preflight.json` |
| safetune | ok | 110.2 | `results/d1/smoke/safetune.json` |
| circuitkit | failed |  | `results/d1/smoke/circuitkit.json` |
| auditkit | ok | 56.3 | `results/d1/smoke/auditkit.json` |
| curatorkit | ok | 0.3 | `results/d1/smoke/curatorkit.json` |

## Model access

- `CohereLabs/tiny-aya-global`: ok
- `CohereLabs/tiny-aya-base`: ok
