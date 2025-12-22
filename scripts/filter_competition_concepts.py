#!/usr/bin/env python3
"""
Create a trimmed concept list containing only the competition labels.

Usage:
    python scripts/filter_competition_concepts.py \
        --concepts concepts/chexpert_concepts.json \
        --output concepts/competition_concepts.txt
"""

import argparse
import json
from pathlib import Path

COMPETITION_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Extract competition concepts")
    parser.add_argument(
        "--concepts",
        type=Path,
        required=True,
        help="Path to the structured concepts JSON"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file (one concept per line)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with args.concepts.open() as f:
        data = json.load(f)

    concepts_dict = data.get("concepts", data)
    filtered = []
    for label in COMPETITION_LABELS:
        entries = concepts_dict.get(label)
        if not entries:
            continue
        filtered.extend(entries)

    args.output.parent.mkdir(exist_ok=True, parents=True)
    with args.output.open("w") as f:
        for concept in filtered:
            f.write(f"{concept}\n")

    print(f"Wrote {len(filtered)} concepts to {args.output}")


if __name__ == "__main__":
    main()
