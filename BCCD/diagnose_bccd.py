"""
Diagnostic: inspect what's actually inside your Annotations/*.xml files and labels.csv,
without doing any conversion. Run this to figure out why only 1 class showed up.

Usage:
    python diagnose_bccd.py --dataset-dir "dataset-master/dataset-master"
"""

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def image_index_from_filename(name: str):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    ann_dir = dataset_dir / "Annotations"
    csv_path = dataset_dir / "labels.csv"

    xml_files = sorted(ann_dir.glob("*.xml"))
    print(f"Found {len(xml_files)} XML files in {ann_dir}\n")

    raw_label_counts = Counter()
    tag_used_counts = Counter()   # did we find label via <n> or <name>?
    per_file_object_count = Counter()

    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        n_objects = 0
        for obj in root.findall("object"):
            n_tag = obj.find("n")
            name_tag = obj.find("name")
            if n_tag is not None and n_tag.text:
                raw_label_counts[n_tag.text.strip()] += 1
                tag_used_counts["<n>"] += 1
            elif name_tag is not None and name_tag.text:
                raw_label_counts[name_tag.text.strip()] += 1
                tag_used_counts["<name>"] += 1
            else:
                raw_label_counts["<<NO LABEL FOUND>>"] += 1
                tag_used_counts["none"] += 1
            n_objects += 1
        per_file_object_count[n_objects] += 1

    print("=== Raw label counts across all XML files ===")
    for label, count in raw_label_counts.most_common():
        print(f"  {label}: {count}")

    print("\n=== Which XML tag held the label ===")
    for tag, count in tag_used_counts.most_common():
        print(f"  {tag}: {count}")

    print("\n=== Objects per file (n_objects: n_files) ===")
    for n_obj, n_files in sorted(per_file_object_count.items()):
        print(f"  {n_obj} objects: {n_files} files")

    # CSV check
    print(f"\n=== labels.csv check ===")
    if not csv_path.exists():
        print(f"  labels.csv NOT FOUND at {csv_path}")
        return

    csv_rows = 0
    empty_category = 0
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        print(f"  CSV columns detected: {reader.fieldnames}")
        for row in reader:
            csv_rows += 1
            if not (row.get("Category") or "").strip():
                empty_category += 1
    print(f"  Total CSV rows: {csv_rows}")
    print(f"  Rows with empty Category: {empty_category}")

    # Cross-check: how many image indices from filenames actually exist as CSV rows?
    csv_indices = set()
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Image", "").strip() != "":
                csv_indices.add(int(row["Image"]))

    xml_indices = set()
    for xml_path in xml_files:
        idx = image_index_from_filename(xml_path.stem)
        if idx is not None:
            xml_indices.add(idx)

    missing = xml_indices - csv_indices
    print(f"\n  XML image indices: {len(xml_indices)} (range {min(xml_indices)}-{max(xml_indices)})")
    print(f"  CSV image indices: {len(csv_indices)} (range {min(csv_indices)}-{max(csv_indices)})")
    print(f"  XML indices with NO matching CSV row: {len(missing)}")
    if missing and len(missing) <= 20:
        print(f"  Missing indices: {sorted(missing)}")


if __name__ == "__main__":
    main()