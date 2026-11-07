"""Generate the self-contained Google Colab D1 G0 notebook."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
script_path = REPO_ROOT / "scripts" / "d1_smoke.py"
script_content = script_path.read_text(encoding="utf-8")

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# CircuitGraft - D1 / G0 Smoke Test on Tiny Aya (Colab GPU)\n",
                "\n",
                "This notebook is **100% self-contained**: it installs the four toolkits, creates `scripts/d1_smoke.py`, runs the checks on `CohereLabs/tiny-aya-global` via CUDA, and packages the G0 evidence for download.\n",
                "\n",
                "> **Prerequisites:** Ensure your Hugging Face account has accepted the licence for `CohereLabs/tiny-aya-global` and `CohereLabs/tiny-aya-base`.",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 1. Verify GPU and Configure Memory (Swapfile)"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!nvidia-smi\n",
                "# Allocate swap on Colab disk to prevent Linux OOM-killer during 3.5B weight processing\n",
                "!fallocate -l 10G /swapfile 2>/dev/null && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile || true\n",
                "!free -h",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2. Install Required Toolkits (SafeTune, CircuitKIT, AuditKIT, CuratorKIT)"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install --upgrade pip\n",
                '!pip install safetune==0.1.12 circuitkit==0.2.0 auditkit==1.1.3 "curatorkit[generation,hf]==0.1.22"\n',
                "!pip install torch transformers transformer_lens peft trl sentence_transformers litellm python-dotenv typer rich\n",
                'print("Dependencies installed successfully!")',
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 3. Set Up Smoke Test Script (`scripts/d1_smoke.py`)"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                'os.makedirs("scripts", exist_ok=True)\n',
                "smoke_code = " + repr(script_content) + "\n",
                'with open("scripts/d1_smoke.py", "w", encoding="utf-8") as f:\n',
                "    f.write(smoke_code)\n",
                'print("Created scripts/d1_smoke.py successfully! Length:", len(smoke_code))',
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 4. Set Hugging Face Token"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "from getpass import getpass\n",
                "try:\n",
                "    from google.colab import userdata\n",
                '    token = userdata.get("HF_TOKEN")\n',
                "except Exception:\n",
                "    token = None\n",
                "if not token:\n",
                '    token = getpass("Enter your Hugging Face token: ")\n',
                'os.environ["HF_TOKEN"] = token.strip()\n',
                'os.environ["HF_ORG"] = "dubeydevansh"\n',
                'print("HF_TOKEN set successfully!")',
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 5. Run D1 / G0 Smoke Test on CUDA (Real Tiny Aya)"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python scripts/d1_smoke.py --device cuda --output-dir results/d1/smoke --skip-existing"
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 6. Inspect Summary and G0 Gate Status"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "from IPython.display import Markdown, display\n",
                'summary_path = "results/d1/smoke/summary.md"\n',
                "if os.path.exists(summary_path):\n",
                '    with open(summary_path, "r", encoding="utf-8") as f:\n',
                "        display(Markdown(f.read()))\n",
                "else:\n",
                '    print("Error: summary.md not found. Checking outputs in results/d1/smoke:")\n',
                "    !ls -la results/d1/smoke\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 7. Download G0 Evidence Archive"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!zip -r d1_smoke_results.zip results/d1/smoke\n",
                "try:\n",
                "    from google.colab import files\n",
                '    files.download("d1_smoke_results.zip")\n',
                "except Exception as e:\n",
                '    print("Archive available at d1_smoke_results.zip:", e)',
            ],
        },
    ],
    "metadata": {
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 2,
}

out_path = REPO_ROOT / "notebooks" / "run_d1_g0_smoke.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Wrote self-contained notebook to {out_path} ({out_path.stat().st_size} bytes)")
