"""
query_processor.py

Dummy stub for query preprocessing: tokenization, normalization,
stopword removal, spell correction, query expansion, etc.
"""

import re
from typing import List

STOPWORDS = {"a", "an", "the", "is", "are", "of", "and", "to", "in"}


class QueryProcessor:
    def __init__(self, remove_stopwords: bool = True):
        self.remove_stopwords = remove_stopwords

    def tokenize(self, query: str) -> List[str]:
        """TODO: replace with a real tokenizer (spaCy/NLTK/subword)."""
        tokens = re.findall(r"[a-z0-9]+", query.lower())
        return tokens

    def normalize(self, tokens: List[str]) -> List[str]:
        """TODO: add stemming/lemmatization here."""
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]
        return tokens

    def process(self, query: str) -> List[str]:
        tokens = self.tokenize(query)
        tokens = self.normalize(tokens)
        return tokens


if __name__ == "__main__":
    qp = QueryProcessor()
    processed = qp.process("What is the best search engine for research papers?")
    print("[stub] Processed query:", processed)
