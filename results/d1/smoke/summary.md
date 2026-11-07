# D1 smoke summary

- Generated: 2026-11-07T16:55:22.308694+00:00
- Mode: Tiny Aya
- Model: `CohereLabs/tiny-aya-global`
- G0 target: `CohereLabs/tiny-aya-global`
- Device: `cuda` (bfloat16)
- G0: **passed** - all four toolkits ran on Tiny Aya

| Check | Status | Wall clock (s) | Log |
|---|---|---|---|
| preflight | ok | 19.9 | `results/d1/smoke/preflight.json` |
| safetune | ok | 52.7 | `results/d1/smoke/safetune.json` |
| circuitkit | ok | 69.8 | `results/d1/smoke/circuitkit.json` |
| auditkit | ok | 0.2 | `results/d1/smoke/auditkit.json` |
| curatorkit | ok | 0.4 | `results/d1/smoke/curatorkit.json` |

## Model access

- `CohereLabs/tiny-aya-global`: ok
- `CohereLabs/tiny-aya-base`: ok
