# D1 smoke summary

- Generated: 2026-11-05T17:11:18.244746+00:00
- Mode: proxy
- Model: `HuggingFaceTB/SmolLM-135M-Instruct`
- G0 target: `CohereLabs/tiny-aya-global`
- Device: `cpu` (float32)
- G0: **not_evaluated** - proxy mode passed; G0 requires the real Tiny Aya checkpoints

| Check | Status | Wall clock (s) | Log |
|---|---|---|---|
| preflight | blocked | 11.3 | `results\d1\smoke_proxy\preflight.json` |
| safetune | ok | 14.3 | `results\d1\smoke_proxy\safetune.json` |
| circuitkit | ok | 25.8 | `results\d1\smoke_proxy\circuitkit.json` |
| auditkit | ok | 10.6 | `results\d1\smoke_proxy\auditkit.json` |
| curatorkit | ok | 0.1 | `results\d1\smoke_proxy\curatorkit.json` |

## Model access

- `CohereLabs/tiny-aya-global`: blocked - 403 Client Error. (Request ID: Root=1-6aecb874-6f90cd594238d2f20bfacd03;9e8a57f5-f71e-495e-b409-ecf0f6a23758)

Cannot access gated repo for url https://huggingface.co/CohereLabs/tiny-aya-global/resolve/main/config.json.
Access to model CohereLabs/tiny-aya-global is restricted and you are not in the authorized list. Visit https://huggingface.co/CohereLabs/tiny-aya-global to ask for access.
- `CohereLabs/tiny-aya-base`: blocked - 403 Client Error. (Request ID: Root=1-6aecb874-6ee2116341aba5f85f4971ce;f92316ab-a77d-463a-9622-9a62ac47c3a5)

Cannot access gated repo for url https://huggingface.co/CohereLabs/tiny-aya-base/resolve/main/config.json.
Access to model CohereLabs/tiny-aya-base is restricted and you are not in the authorized list. Visit https://huggingface.co/CohereLabs/tiny-aya-base to ask for access.
