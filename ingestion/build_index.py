"""
build_index.py

Dummy stub for building an inverted index (or vector index) from
cleaned documents in data/processed/.
"""

import argparse
import json
import os
import pickle
from collections import defaultdict


DEFAULT_INPUT = os.path.join("data", "processed", "clean_data.jsonl")
DEFAULT_INDEX_PATH = os.path.join("data", "processed", "inverted_index.pkl")


def build_inverted_index(input_path: str = DEFAULT_INPUT):
    """
    Placeholder inverted index: term -> list of doc_ids.
    Replace with a real tokenizer + posting list structure
    (and eventually a proper index like Lucene/Elasticsearch/FAISS).
    """
    index = defaultdict(list)
    doc_store = {}

    if not os.path.exists(input_path):
        print(f"[stub] Input file not found: {input_path} (building empty index)")
        return index, doc_store

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc_id = doc.get("doc_id")
            text = doc.get("text", "")
            doc_store[doc_id] = doc

            for token in text.split():
                index[token].append(doc_id)

    return index, doc_store


def save_index(index, doc_store, output_path: str = DEFAULT_INDEX_PATH):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"index": dict(index), "doc_store": doc_store}, f)
    print(f"[stub] Saved index with {len(index)} terms, {len(doc_store)} docs -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build inverted index from cleaned data.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_INDEX_PATH)
    args = parser.parse_args()

    index, doc_store = build_inverted_index(args.input)
    save_index(index, doc_store, args.output)


if __name__ == "__main__":
    main()
