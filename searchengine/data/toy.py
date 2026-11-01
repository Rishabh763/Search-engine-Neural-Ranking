"""A small synthetic collection with known structure, for tests and offline smoke runs.

Each document belongs to one topic and draws ~40% of its words from that topic's
vocabulary, the rest from a Zipf-distributed background vocabulary. Queries are
3 topic words. Graded labels: 2 if a same-topic doc contains all query words,
1 if it contains at least one. Lexical retrieval should do clearly better than
random here, which is what the smoke test checks.
"""

from __future__ import annotations

import random

from searchengine.types import Document, Qrels, Queries


def make_toy_collection(
    n_docs: int = 2000, n_topics: int = 40, n_queries: int = 80, doc_len: int = 40, seed: int = 7
) -> tuple[list[Document], Queries, Qrels]:
    rng = random.Random(seed)
    syll = ["ka", "lo", "mi", "ne", "ru", "ta", "vo", "zi", "pe", "su", "do", "fa"]

    def word() -> str:
        return "".join(rng.choice(syll) for _ in range(rng.randint(2, 4)))

    background = list(dict.fromkeys(word() for _ in range(3000)))
    topics = [list(dict.fromkeys(word() + "x" for _ in range(25))) for _ in range(n_topics)]
    zipf_w = [1.0 / (i + 1) for i in range(len(background))]

    docs: list[Document] = []
    doc_topic: dict[str, int] = {}
    for i in range(n_docs):
        t = i % n_topics
        n_topic_words = int(doc_len * 0.4)
        words = rng.choices(topics[t], k=n_topic_words) + rng.choices(background, weights=zipf_w, k=doc_len - n_topic_words)
        rng.shuffle(words)
        did = f"D{i}"
        docs.append(Document(did, " ".join(words)))
        doc_topic[did] = t

    queries: Queries = {}
    qrels: Qrels = {}
    doc_words = {d.id: set(d.text.split()) for d in docs}
    for qi in range(n_queries):
        t = qi % n_topics
        qwords = rng.sample(topics[t], 3)
        qid = f"Q{qi}"
        queries[qid] = " ".join(qwords)
        rels = {}
        for d in docs:
            if doc_topic[d.id] != t:
                continue
            overlap = len(set(qwords) & doc_words[d.id])
            if overlap == 3:
                rels[d.id] = 2
            elif overlap >= 1:
                rels[d.id] = 1
        if any(r > 0 for r in rels.values()):
            qrels[qid] = rels
        else:
            del queries[qid]
    return docs, queries, qrels
