"""Compare the local analyzer with Elasticsearch's, token for token, on real text.

Sends a sample of passages and every query through ES's `_analyze` API (using the
index's own `contents` field analyzer, so it's exactly what was indexed) and
through searchengine.text.Analyzer, then reports:
  - % of texts whose token sequences differ
  - the tokens that most often appear on only one side, with an example each

python scripts/check_analyzer.py --collection msmarco-100k
python scripts/check_analyzer.py --collection msmarco-100k --tokenizer v1   # the old tokenizer
"""

import argparse
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import _bootstrap  # noqa: F401

from searchengine.config import get_settings
from searchengine.data.collection import Collection
from searchengine.index.es import SEARCH_FIELD, make_client
from searchengine.text import DEFAULT_TOKENIZER, Analyzer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", required=True)
    p.add_argument("--index-name", default=None)
    p.add_argument("--n-docs", type=int, default=2000)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    s = get_settings()
    coll = Collection.from_name(args.collection, s.data_dir)
    index = args.index_name or coll.name
    es = make_client(s.es_url)

    # Reservoir sample of passages (one streaming pass, O(n) memory).
    rng = random.Random(args.seed)
    sample: list[str] = []
    for i, d in enumerate(coll.iter_docs()):
        text = f"{d.title} {d.text}" if d.title else d.text
        if i < args.n_docs:
            sample.append(text)
        else:
            j = rng.randint(0, i)
            if j < args.n_docs:
                sample[j] = text
    queries = [q for es_name in coll.eval_set_names for q in coll.queries(es_name).values()]
    texts = [("doc", t) for t in sample] + [("query", q) for q in queries]

    def es_tokens(text: str) -> list[str]:
        r = es.indices.analyze(index=index, field=SEARCH_FIELD, text=text)
        return [t["token"] for t in r["tokens"]]

    # Parallel requests: each call is dominated by HTTP round-trip time, not work.
    with ThreadPoolExecutor(args.workers) as pool:
        es_out = list(pool.map(es_tokens, [t for _, t in texts]))

    analyzer = Analyzer(tokenizer=args.tokenizer)
    mismatched = {"doc": 0, "query": 0}
    totals = Counter(kind for kind, _ in texts)
    only_es, only_local = Counter(), Counter()
    example: dict[tuple[str, str], str] = {}
    n_tok_es = n_tok_local = 0
    for (kind, text), ref in zip(texts, es_out):
        ours = analyzer(text)
        n_tok_es += len(ref)
        n_tok_local += len(ours)
        if ours != ref:
            mismatched[kind] += 1
            a, b = Counter(ref), Counter(ours)
            for tok, c in (a - b).items():
                only_es[tok] += c
                example.setdefault(("es", tok), text)
            for tok, c in (b - a).items():
                only_local[tok] += c
                example.setdefault(("local", tok), text)

    print(f"tokenizer={args.tokenizer}  index={index}")
    for kind in ("doc", "query"):
        n = totals[kind]
        print(f"  {kind:5s}: {mismatched[kind]}/{n} texts differ ({100 * mismatched[kind] / max(n, 1):.1f}%)")
    print(f"  total tokens: ES={n_tok_es:,}  local={n_tok_local:,}  (diff {n_tok_local - n_tok_es:+,})")

    def show(title, counter, side):
        print(f"\n{title}")
        for tok, c in counter.most_common(15):
            ex = example[(side, tok)]
            i = ex.lower().find(tok[:4])
            snippet = ex[max(0, i - 30) : i + 40] if i >= 0 else ex[:70]
            print(f"  {c:6d}  {tok!r:22s}  ...{snippet.strip()}...")

    show("Tokens only Elasticsearch produced:", only_es, "es")
    show("Tokens only the local analyzer produced:", only_local, "local")


if __name__ == "__main__":
    main()
