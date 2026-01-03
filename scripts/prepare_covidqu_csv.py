#!/usr/bin/env python3
"""
Prepare train/val/test CSVs for the COVID-QU dataset.

Expected directory structure (default: Infection Segmentation Data):
<dataset_root>/
  Infection Segmentation Data/Infection Segmentation Data/
    Train/{COVID-19,Non-COVID,Normal}/images/*.png|jpg
    Val/{...}/images
    Test/{...}/images

Outputs:
  train.csv, valid.csv, test.csv in --output_dir (default: --dataset_root)
Columns:
  Path (relative to dataset_root), COVID-19, Non-COVID, Normal (one-hot)

You can point --variant lung to use the Lung Segmentation Data tree instead.
"""

import argparse
import csv
import os
from typing import Dict, List


CLASS_NAMES = ["COVID-19", "Non-COVID", "Normal"]
VARIANT_DIRS = {
    "infection": os.path.join("Infection Segmentation Data", "Infection Segmentation Data"),
    "lung": os.path.join("Lung Segmentation Data", "Lung Segmentation Data"),
}


def collect_split(root: str, split: str, variant_dir: str) -> List[Dict[str, str]]:
    records = []
    for cls in CLASS_NAMES:
        img_dir = os.path.join(root, variant_dir, split, cls, "images")
        if not os.path.isdir(img_dir):
            print(f"Warning: missing directory {img_dir}, skipping.")
            continue
        for fname in os.listdir(img_dir):
            if fname.startswith("."):
                continue
            full_path = os.path.join(img_dir, fname)
            rel_path = os.path.relpath(full_path, root)
            row = {"Path": rel_path}
            for name in CLASS_NAMES:
                row[name] = 1 if name == cls else 0
            records.append(row)
    return records


def write_csv(path: str, rows: List[Dict[str, str]]) -> None:
    if not rows:
        print(f"Warning: no rows to write for {path}")
        return
    fieldnames = ["Path"] + CLASS_NAMES
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate COVID-QU CSVs for CBM training")
    parser.add_argument("--dataset_root", required=True, help="Root folder containing the COVID-QU dataset")
    parser.add_argument("--output_dir", default=None, help="Where to write CSVs (default: dataset_root)")
    parser.add_argument("--variant", choices=["infection", "lung"], default="infection",
                        help="Which COVID-QU variant to use")
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    output_dir = os.path.abspath(args.output_dir or dataset_root)
    os.makedirs(output_dir, exist_ok=True)

    variant_dir = VARIANT_DIRS[args.variant]
    print(f"Using variant directory: {variant_dir}")

    split_map = {"train": "Train", "valid": "Val", "test": "Test"}
    for split_key, split_dir in split_map.items():
        rows = collect_split(dataset_root, split_dir, variant_dir)
        out_path = os.path.join(output_dir, f"{'valid' if split_key == 'valid' else split_key}.csv")
        write_csv(out_path, rows)

    print("Done. Pass --label_set covidqu and --data_dir <output_dir> to the CBM scripts.")


if __name__ == "__main__":
    main()
