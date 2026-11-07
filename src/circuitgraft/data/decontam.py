"""Decontamination: 13-gram overlap plus optional embedding-cosine filtering."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def word_ngrams(text: str, n: int = 13) -> set[tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be positive")
    words = normalize_text(text).split()
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def ngram_index(references: Sequence[str], n: int = 13) -> set[tuple[str, ...]]:
    """Build one lookup set for all reference texts.

    ``find_ngram_overlap`` recomputes every reference's n-grams for every item,
    which is quadratic on a 20k-row SFT corpus. The eval decontamination step
    uses this index instead.
    """
    index: set[tuple[str, ...]] = set()
    for reference in references:
        index.update(word_ngrams(reference, n))
    return index


def find_ngram_overlap_indexed(
    text: str, index: set[tuple[str, ...]], n: int = 13
) -> str | None:
    """Return a matched n-gram (joined) from a precomputed reference index."""
    shared = word_ngrams(text, n) & index
    if shared:
        return " ".join(sorted(shared)[0])
    return None


def find_ngram_overlap(text: str, references: Sequence[str], n: int = 13) -> str | None:
    """Return a matched n-gram (joined) or None."""
    return find_ngram_overlap_indexed(text, ngram_index(references, n), n)


@dataclass
class DecontamReport:
    kept: list[dict[str, Any]] = field(default_factory=list)
    removed: list[dict[str, Any]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    ngram_size: int = 13
    embedding_threshold: float | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "kept": len(self.kept),
            "removed": len(self.removed),
            "ngram_size": self.ngram_size,
            "embedding_threshold": self.embedding_threshold,
            "reasons": self.reasons[:50],
        }


def decontaminate_by_ngram(
    items: Iterable[dict[str, Any]],
    references: Sequence[str],
    *,
    text_key: str = "text",
    n: int = 13,
) -> DecontamReport:
    report = DecontamReport(ngram_size=n)
    for item in items:
        text = str(item.get(text_key, ""))
        match = find_ngram_overlap(text, references, n)
        if match is None:
            report.kept.append(item)
        else:
            report.removed.append(item)
            report.reasons.append(f"{item.get('id', '<no-id>')}: {n}-gram: {match[:120]}")
    return report


def decontaminate_by_embedding(
    items: Iterable[dict[str, Any]],
    references: Sequence[str],
    *,
    text_key: str = "text",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    threshold: float = 0.9,
    batch_size: int = 64,
) -> DecontamReport:
    """Drop items whose embedding cosine similarity to any reference exceeds threshold.

    This is an intentional O(n·m) scan over the (small) eval sets; it is not used
    on the full SFT corpus.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    report = DecontamReport(embedding_threshold=threshold)
    if not references:
        report.kept = list(items)
        return report

    model = SentenceTransformer(model_name)
    reference_embeddings = model.encode(
        list(references), convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size
    )
    kept: list[dict[str, Any]] = []
    for item in items:
        text = str(item.get(text_key, ""))
        embedding = model.encode([text], convert_to_tensor=True, normalize_embeddings=True)[0]
        similarity = float(torch.max(reference_embeddings @ embedding).item())
        if similarity > threshold:
            report.removed.append(item)
            report.reasons.append(f"{item.get('id', '<no-id>')}: cosine {similarity:.3f}")
        else:
            kept.append(item)
    report.kept = kept
    return report


def decontaminate(
    items: Iterable[dict[str, Any]],
    references: Sequence[str],
    *,
    text_key: str = "text",
    n: int = 13,
    use_embeddings: bool = False,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_threshold: float = 0.9,
) -> DecontamReport:
    items = list(items)
    ngram_report = decontaminate_by_ngram(items, references, text_key=text_key, n=n)
    if not use_embeddings:
        return ngram_report
    embedding_report = decontaminate_by_embedding(
        ngram_report.kept,
        references,
        text_key=text_key,
        model_name=embedding_model,
        threshold=embedding_threshold,
    )
    embedding_report.removed = ngram_report.removed + embedding_report.removed
    embedding_report.reasons = ngram_report.reasons + embedding_report.reasons
    embedding_report.ngram_size = n
    return embedding_report
