"""
clean_data.py

Dummy stub for cleaning/normalizing raw documents before indexing.
Reads from data/raw/, writes cleaned docs to data/processed/.
"""

import argparse
import json
import os
import re


DEFAULT_INPUT = os.path.join("data", "raw", "raw_data.jsonl")
DEFAULT_OUTPUT = os.path.join("data", "processed", "clean_data.jsonl")


def clean_text(text: str) -> str:
    """
    Basic placeholder cleaning: lowercase + collapse whitespace.
    Replace with real normalization (HTML stripping, unicode
    normalization, boilerplate removal, etc.).
    """
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def clean_data(input_path: str = DEFAULT_INPUT, output_path: str = DEFAULT_OUTPUT) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_docs = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        if os.path.exists(input_path):
            with open(input_path, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    doc = json.loads(line)
                    doc["text"] = clean_text(doc.get("text", ""))
                    out_f.write(json.dumps(doc) + "\n")
                    n_docs += 1
        else:
            print(f"[stub] Input file not found: {input_path} (skipping)")

    print(f"[stub] Cleaned {n_docs} documents -> {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Clean raw documents.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    clean_data(args.input, args.output)


if __name__ == "__main__":
    main()
