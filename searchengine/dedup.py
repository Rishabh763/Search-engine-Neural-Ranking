"""Near-duplicate collapsing for result lists.

MS MARCO (like the web it was scraped from) holds many copies of the same passage:
the same text from different URLs, re-cased, stripped of punctuation, or cut short.
Both neural models score copies alike, so without collapsing a user can see one
answer four times in the top five.

Test: word 3-shingles (over lowercase alphanumeric tokens, so case, punctuation and
spacing don't matter) compared by *containment*,

    containment(A, B) = |A ∩ B| / min(|A|, |B|),

not Jaccard |A ∩ B| / |A ∪ B|. A truncated copy (the same passage cut after two of
its three sentences) has nearly all of its shingles inside the full one, so its
containment is ~1 while its Jaccard is ~2/3. Two passages count as duplicates when
containment >= threshold (default 0.8).

Collapsing is greedy in rank order: each result is compared with the results
already kept; a duplicate is attached to the first kept result it matches and
dropped from the list. The best-ranked copy therefore represents its group, and
the cost is at most (#candidates x #kept) set intersections: a few milliseconds
for 100 candidates and k = 10.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_TOKEN = re.compile(r"[a-z0-9]+")
DEFAULT_THRESHOLD = 0.8


def shingles(text: str, n: int = 3) -> frozenset:
    """Word n-grams over lowercase alphanumeric tokens. Texts shorter than n words
    give one shingle (the whole token tuple); an empty text gives an empty set."""
    toks = _TOKEN.findall(text.lower())
    if len(toks) < n:
        return frozenset([tuple(toks)]) if toks else frozenset()
    return frozenset(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))


def containment(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    return sum(1 for s in small if s in large) / len(small)


@dataclass
class Collapsed:
    kept: list[int]  # indices into the input, in input (rank) order
    duplicates: dict[int, list[int]] = field(default_factory=dict)  # kept index -> dropped indices
    examined: int = 0  # how many inputs were looked at before stopping

    @property
    def n_dropped(self) -> int:
        return sum(len(v) for v in self.duplicates.values())


def collapse(texts: list[str], k: int | None = None, threshold: float = DEFAULT_THRESHOLD, n: int = 3) -> Collapsed:
    """Greedy near-duplicate collapsing of a ranked list of texts.

    Stops once `k` results are kept (None = process everything). Returns which
    indices to keep and, for each kept one, the indices collapsed into it."""
    out = Collapsed(kept=[])
    kept_sh: list[frozenset] = []
    for i, t in enumerate(texts):
        if k is not None and len(out.kept) >= k:
            break
        out.examined = i + 1
        sh = shingles(t, n)
        match = next((j for j, ks in enumerate(kept_sh) if containment(sh, ks) >= threshold), None)
        if match is None:
            out.kept.append(i)
            kept_sh.append(sh)
        else:
            out.duplicates.setdefault(out.kept[match], []).append(i)
    return out
