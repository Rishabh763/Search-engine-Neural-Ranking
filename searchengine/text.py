"""Text analysis for lexical retrieval.

Mirrors Elasticsearch's built-in `english` analyzer closely enough that the local
BM25 and the Elasticsearch BM25 should produce near-identical rankings:

    ES english analyzer:  standard tokenizer -> possessive 's removal -> lowercase
                          -> English stopwords (Lucene's 33) -> Porter stemmer
    this module:          regex word tokenizer -> lowercase -> possessive 's removal
                          -> same 33 stopwords -> Porter stemmer (PyStemmer)

Known gap: Lucene's standard tokenizer follows Unicode UAX#29 word boundaries,
which treats a few cases differently (e.g. "3.5" stays one token there).

Note: this analysis is for BM25 only. Neural models get raw text: stemming and
stopword removal destroy the input distribution transformers were trained on.
"""

from __future__ import annotations

import re

# Lucene EnglishAnalyzer.ENGLISH_STOP_WORDS_SET (what ES `_english_` uses).
ENGLISH_STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such "
    "that the their then there these they this to was will with".split()
)

_TOKEN_RE = re.compile(r"\w+(?:'\w+)*")


class Analyzer:
    def __init__(self, stopwords: frozenset[str] | None = ENGLISH_STOPWORDS, stemmer: str | None = "porter"):
        self.stopwords = stopwords or frozenset()
        self.stemmer_name = stemmer
        self._stemmer = None
        if stemmer:
            import Stemmer  # PyStemmer: C implementation, ~100x faster than nltk

            self._stemmer = Stemmer.Stemmer(stemmer)

    def __call__(self, text: str) -> list[str]:
        toks = _TOKEN_RE.findall(text.lower())
        toks = [t[:-2] if t.endswith("'s") else t for t in toks]
        toks = [t for t in toks if t and t not in self.stopwords]
        if self._stemmer is not None:
            toks = self._stemmer.stemWords(toks)
        return toks

    def config(self) -> dict:
        return {"stopwords": "lucene_english" if self.stopwords else None, "stemmer": self.stemmer_name}

    @classmethod
    def from_config(cls, cfg: dict) -> "Analyzer":
        return cls(ENGLISH_STOPWORDS if cfg.get("stopwords") else None, cfg.get("stemmer"))
