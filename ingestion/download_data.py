"""
download_data.py

Dummy stub for downloading raw corpus data into data/raw/.
Replace the placeholder logic with real dataset download code
(e.g. MS MARCO, Wikipedia dump, custom crawl, etc.).
"""

import argparse
import os


DEFAULT_OUTPUT_DIR = os.path.join("data", "raw")


def download_data(source_url: str, output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    """
    Download raw data from `source_url` into `output_dir`.

    TODO: implement actual download logic (requests/wget/boto3/etc.)
    """
    os.makedirs(output_dir, exist_ok=True)
    dest_path = os.path.join(output_dir, "raw_data.jsonl")

    print(f"[stub] Would download from: {source_url}")
    print(f"[stub] Would save to: {dest_path}")

    # Placeholder: write an empty file so downstream steps don't break.
    with open(dest_path, "w", encoding="utf-8") as f:
        pass

    return dest_path


def main():
    parser = argparse.ArgumentParser(description="Download raw corpus data.")
    parser.add_argument("--source-url", type=str, default="https://example.com/dataset.jsonl")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    download_data(args.source_url, args.output_dir)


if __name__ == "__main__":
    main()
