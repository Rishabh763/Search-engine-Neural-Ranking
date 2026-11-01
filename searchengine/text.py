"""Text analysis for lexical retrieval.

Mirrors Elasticsearch's built-in `english` analyzer so the local BM25 and the
Elasticsearch BM25 index exactly the same terms:

    ES english analyzer:  standard tokenizer (Unicode UAX#29 word boundaries)
                          -> possessive filter (strips 's, ’s, ＇s) -> lowercase
                          -> English stopwords (Lucene's 33) -> Lucene PorterStemmer
    this module:          regex approximation of UAX#29 -> lowercase
                          -> same possessive rule -> same stopwords -> matching Porter

Versions are stored in each index's metadata, so an index is always queried with
the analyzer that built it. Each step was driven by scripts/check_analyzer.py,
which diffs our tokens against ES's `_analyze` on real MS MARCO text:

    v1  \\w+ joined across ASCII apostrophes, Snowball Porter.         41.5% of passages differed
        Curly apostrophes (U+2019, everywhere in MS MARCO) split "runner’s" into
        "runner" + "s", and Snowball stems "s" to "": empty terms in the index.
    v2  UAX#29 joins (. ' ’ ＇ between letters, . , ' between digits);  6.5% differed
        never stem words of 1-2 letters (Lucene's guard: "us" stays "us").
    v3  Lucene's actual stemmer: Martin Porter's reference implementation, which
        departs from the published algorithm ("logi"->"log", "bli"->"ble"), so
        "technology" -> "technolog" (Snowball: "technologi"). Joins require the
        same character class on both sides ("No.1" -> "no" + "1"); pictographic
        symbols (® ™ © ♠) are tokens; underscore-only "words" are dropped.

Remaining known gaps (rare in English web text): ':' inside words, CJK ideographs
(Lucene emits one token per character).

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

_APOS = "'’＇"  # ASCII, right single quotation mark, fullwidth apostrophe
_LETTER = r"[^\W\d_]"
# Common Extended_Pictographic code points; Lucene's tokenizer emits each as a token.
_PICTO = (
    "©®‼⁉™ℹ↔-↙↩↪⌚⌛⌨⏏"
    "⏩-⏳⏸-⏺Ⓜ▪▫▶◀◻-◾☀-➿"
    "⤴⤵⬅-⬇⬛⬜⭐⭕〰〽㊗㊙"
    "\U0001f000-\U0001faff"
)

TOKENIZERS = {
    "v1": re.compile(r"\w+(?:'\w+)*"),
    "v2": re.compile(rf"\w+(?:(?:[.{_APOS}]|(?<=\d),(?=\d))\w+)*"),
    "v3": re.compile(
        rf"\w+(?:(?:(?<={_LETTER})[.{_APOS}](?={_LETTER})|(?<=\d)[.,{_APOS}](?=\d))\w+)*"
        rf"|[{_PICTO}]"
    ),
}
DEFAULT_TOKENIZER = "v3"
DEFAULT_STEMMER = "porter_lucene"
_POSSESSIVE = tuple(a + "s" for a in _APOS)


class _LucenePorter:
    """Martin Porter's reference Porter stemmer (what Lucene ships), memoized.

    NLTK's MARTIN_EXTENSIONS mode implements exactly that reference code, including
    its departures from the paper and the "don't stem 1-2 letter words" guard. It
    is pure Python, so results are cached per word: a corpus has far fewer distinct
    words than tokens (MS MARCO 100K: ~120K distinct vs ~5M tokens).
    """

    def __init__(self, max_cache: int = 2_000_000):
        from nltk.stem.porter import PorterStemmer

        self._stem = PorterStemmer(mode=PorterStemmer.MARTIN_EXTENSIONS).stem
        self._cache: dict[str, str] = {}
        self._max = max_cache

    def __call__(self, toks: list[str]) -> list[str]:
        out = []
        cache = self._cache
        for t in toks:
            s = cache.get(t)
            if s is None:
                # to_lowercase=False: input is already lowercased, and NLTK's own
                # lowercasing would differ from ours on a few Unicode characters.
                s = self._stem(t, to_lowercase=False)
                if len(cache) < self._max:
                    cache[t] = s
            out.append(s)
        return out


class Analyzer:
    def __init__(
        self,
        stopwords: frozenset[str] | None = ENGLISH_STOPWORDS,
        stemmer: str | None = DEFAULT_STEMMER,
        tokenizer: str = DEFAULT_TOKENIZER,
    ):
        if tokenizer not in TOKENIZERS:
            raise ValueError(f"unknown tokenizer {tokenizer!r}; known: {sorted(TOKENIZERS)}")
        if stemmer not in (None, "porter", "porter_lucene"):
            raise ValueError(f"unknown stemmer {stemmer!r}")
        self.stopwords = stopwords or frozenset()
        self.stemmer_name = stemmer
        self.tokenizer = tokenizer
        self._token_re = TOKENIZERS[tokenizer]
        self._possessive = ("'s",) if tokenizer == "v1" else _POSSESSIVE
        self._stem = self._make_stemmer(stemmer, tokenizer)

    @staticmethod
    def _make_stemmer(stemmer: str | None, tokenizer: str):
        if stemmer is None:
            return None
        if stemmer == "porter_lucene":
            return _LucenePorter()
        import Stemmer  # PyStemmer (Snowball's "porter": the published algorithm)

        snow = Stemmer.Stemmer("porter")
        if tokenizer == "v1":  # preserved bug-for-bug so v1 indexes still load correctly
            return snow.stemWords
        return lambda toks: [t if len(t) <= 2 else snow.stemWord(t) for t in toks]

    def __call__(self, text: str) -> list[str]:
        toks = self._token_re.findall(text.lower())
        toks = [t[:-2] if t.endswith(self._possessive) else t for t in toks]
        if self.tokenizer == "v3":
            toks = [t for t in toks if t.strip("_")]  # Lucene emits no underscore-only words
        toks = [t for t in toks if t and t not in self.stopwords]
        return self._stem(toks) if self._stem is not None else toks

    def config(self) -> dict:
        return {
            "stopwords": "lucene_english" if self.stopwords else None,
            "stemmer": self.stemmer_name,
            "tokenizer": self.tokenizer,
        }

    @classmethod
    def from_config(cls, cfg: dict) -> "Analyzer":
        # Indexes written before versioning existed used v1 + Snowball porter.
        return cls(ENGLISH_STOPWORDS if cfg.get("stopwords") else None, cfg.get("stemmer"), cfg.get("tokenizer", "v1"))
