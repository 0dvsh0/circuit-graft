# CircuitGraft Space

Read-only results demo for the tracked results bundle. The app is data-driven:
it displays `results/s9/report-*/` tables and plots, plus optional cached preset
responses. It does not load a model, generate text, or invent metrics.

## Deployed (live)

**https://huggingface.co/spaces/dubeydevansh/circuitgraft** — a *static* Space
(free tier). Gradio SDK Spaces now require HF PRO (API returned 402), so the
deployed version is `space/index.html`: a static page that fetches the tracked
bundle from the Space repo at load time (`results/s9/report-1794803399/`,
uploaded verbatim, commit `35a9ca1a`). The Compare tab (cached responses) is
only available in the Gradio variant below.

To update the deployed bundle: re-run `cg report`, then upload the new
`results/s9/report-<id>/` to the Space and change the `RUN_DIR` constant at the
top of `index.html`.

## Gradio variant (local / PRO)

`app.py` + `requirements.txt` in this folder run the same data-driven UI under
Gradio, including the Compare tab:

1. Create a Hugging Face Space with the Gradio SDK (PRO).
2. Copy this folder's contents (`app.py`, `requirements.txt`) to the Space root.
3. Copy the newest tracked bundle into the Space:

   ```text
   results/s9/report-<timestamp>/
     summary.json
     manifest.json
     summary.md
     table_main.md
     table_attacks.md
     table_overlap.md
     pareto_frontier.png
     circuit_overlap_heatmap.png
     attack_decay.png   # optional
   ```

4. Optional: add `cached_responses.json` beside `app.py` for the Compare tab.
   Do **not** include full harmful completions.

`app.py` looks for the bundle in this order:

1. `CIRCUITGRAFT_REPORT_DIR` (exact directory)
2. `CIRCUITGRAFT_RESULTS_DIR` (directory containing `report-*`)
3. `results/s9/` beside the app, then in the parent directory

It picks the newest `report-*` with a `summary.json`. If no bundle exists, the
Space shows setup instructions instead of placeholder numbers.

## Local preview

```bash
pip install -e ".[space]"
python space/app.py
```

The app imports Gradio lazily, so the loader helpers can be tested without the
UI dependency: `pytest tests/test_space_app.py -q`.

## Cached responses schema

```json
{
  "models": ["aligned", "drifted", "resta", "cg_project"],
  "prompts": [
    {
      "id": "harmful-1",
      "label": "Harmful request (refusal check)",
      "category": "harmful",
      "text": "..."
    },
    {
      "id": "med-1",
      "label": "Medical MCQ",
      "category": "domain",
      "text": "..."
    }
  ],
  "responses": {
    "aligned": {"harmful-1": "I can't help with that.", "med-1": "Answer: B."},
    "drifted": {"harmful-1": "...", "med-1": "..."},
    "resta": {"harmful-1": "...", "med-1": "..."},
    "cg_project": {"harmful-1": "...", "med-1": "..."}
  }
}
```

The Compare tab only displays the strings in this file. Treat it as a curated
refusal/domain-answer sheet, not a harmful-content archive.
