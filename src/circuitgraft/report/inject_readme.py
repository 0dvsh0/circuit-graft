"""Safe injection of generated tables and plots into README.md and documentation (S9).

Locates marker pairs such as:
    <!-- RESULTS:MAIN_TABLE_START -->
    ...
    <!-- RESULTS:MAIN_TABLE_END -->
and replaces everything between them with freshly generated content.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

MARKERS = {
    "main_table": ("<!-- RESULTS:MAIN_TABLE_START -->", "<!-- RESULTS:MAIN_TABLE_END -->"),
    "attack_table": ("<!-- RESULTS:ATTACK_TABLE_START -->", "<!-- RESULTS:ATTACK_TABLE_END -->"),
    "overlap_table": ("<!-- RESULTS:OVERLAP_TABLE_START -->", "<!-- RESULTS:OVERLAP_TABLE_END -->"),
    "pareto_plot": ("<!-- RESULTS:PARETO_PLOT_START -->", "<!-- RESULTS:PARETO_PLOT_END -->"),
}


def inject_section(content: str, section_key: str, new_text: str) -> tuple[str, bool]:
    """Inject new_text between MARKERS[section_key] inside content.

    Returns: (modified_content, did_replace)
    """
    if section_key not in MARKERS:
        raise ValueError(f"Unknown section key: {section_key}. Valid keys: {list(MARKERS.keys())}")

    start_tag, end_tag = MARKERS[section_key]
    pattern = re.compile(
        re.escape(start_tag) + r".*?" + re.escape(end_tag),
        re.DOTALL,
    )

    replacement = f"{start_tag}\n\n{new_text.strip()}\n\n{end_tag}"
    if pattern.search(content):
        return pattern.sub(replacement, content), True
    return content, False


def inject_all(
    doc_path: str | Path,
    sections: dict[str, str],
) -> dict[str, bool]:
    """Inject multiple sections into a markdown file.

    sections is a dict mapping section_key -> replacement text.
    Returns status map of which sections were injected.
    """
    path = Path(doc_path)
    if not path.exists():
        logger.warning("Document not found for injection: %s", path)
        return {k: False for k in sections}

    content = path.read_text(encoding="utf-8")
    status: dict[str, bool] = {}

    for key, new_text in sections.items():
        if key in MARKERS:
            content, replaced = inject_section(content, key, new_text)
            status[key] = replaced
        else:
            status[key] = False

    path.write_text(content, encoding="utf-8")
    return status
