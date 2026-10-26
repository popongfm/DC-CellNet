"""
Convert a BCCD-style dataset (Pascal VOC XML annotations + labels.csv classification file)
into YOLO training format, with WBC boxes replaced by their fine-grained subtype
(NEUTROPHIL, EOSINOPHIL, LYMPHOCYTE, MONOCYTE, BASOPHIL) pulled from labels.csv.

Expected input layout:
    dataset-master/
      Annotations/       *.xml   (Pascal VOC, one file per image: BloodImage_00000.xml ...)
      JPEGImages/         *.jpg  (matching images)
      labels.csv          columns: <blank>, Image, Category

Usage:
    python bccd_to_yolo.py \
        --dataset-dir "dataset-master/dataset-master" \
        --out-dir ./yolo_dataset \
        --val-split 0.2

Output:
    yolo_dataset/
      images/train/*.jpg
      images/val/*.jpg
      labels/train/*.txt
      labels/val/*.txt
      data.yaml
      skipped_boxes.log   (WBC boxes that couldn't be assigned a subtype, if any)
"""

import argparse
import csv
import random
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

WBC_LABEL_IN_XML = "WBC"


def load_labels_csv(csv_path):
    """Map image index (int) -> chosen WBC subtype (str) or None."""
    mapping = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_idx = row.get("Image")
            category = (row.get("Category") or "").strip()
            if img_idx is None or img_idx == "":
                continue
            img_idx = int(img_idx)
            if not category:
                mapping[img_idx] = None
                continue
            # Multi-label rows like "NEUTROPHIL, EOSINOPHIL" -> take the first
            first_label = category.split(",")[0].strip()
            mapping[img_idx] = first_label
    return mapping


def image_index_from_filename(name: str):
    """BloodImage_00000 -> 0"""
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else None


def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.findtext("filename")
    size = root.find("size")
    width = float(size.findtext("width"))
    height = float(size.findtext("height"))

    boxes = []
    for obj in root.findall("object"):
        label = obj.findtext("n") or obj.findtext("name")
        bnd = obj.find("bndbox")
        boxes.append({
            "label": label,
            "xmin": float(bnd.findtext("xmin")),
            "ymin": float(bnd.findtext("ymin")),
            "xmax": float(bnd.findtext("xmax")),
            "ymax": float(bnd.findtext("ymax")),
        })
    return filename, width, height, boxes


def build_dataset(dataset_dir, out_dir, val_split, seed=42):
    dataset_dir = Path(dataset_dir)
    ann_dir = dataset_dir / "Annotations"
    img_dir = dataset_dir / "JPEGImages"
    csv_path = dataset_dir / "labels.csv"

    wbc_subtypes = load_labels_csv(csv_path)

    xml_files = sorted(ann_dir.glob("*.xml"))
    if not xml_files:
        raise SystemExit(f"No XML files found in {ann_dir}")

    all_records = []   # (stem, width, height, [ (class_name, xmin, ymin, xmax, ymax), ... ])
    class_names = []   # preserve first-seen order
    skipped = []

    for xml_path in xml_files:
        stem = xml_path.stem
        img_idx = image_index_from_filename(stem)
        filename, width, height, boxes = parse_voc_xml(xml_path)

        resolved_boxes = []
        for b in boxes:
            label = b["label"]
            if label == WBC_LABEL_IN_XML:
                subtype = wbc_subtypes.get(img_idx)
                if subtype:
                    label = subtype
                else:
                    skipped.append(f"{stem}: WBC box skipped (no subtype in labels.csv)")
                    continue
            if label not in class_names:
                class_names.append(label)
            resolved_boxes.append((label, b["xmin"], b["ymin"], b["xmax"], b["ymax"]))

        all_records.append((stem, width, height, resolved_boxes))

    # Split train/val
    random.seed(seed)
    shuffled = all_records[:]
    random.shuffle(shuffled)
    n_val = int(len(shuffled) * val_split)
    val_records = set(r[0] for r in shuffled[:n_val])

    class_to_id = {name: i for i, name in enumerate(class_names)}

    out_dir = Path(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for stem, width, height, boxes in all_records:
        split = "val" if stem in val_records else "train"

        src_img = img_dir / f"{stem}.jpg"
        if src_img.exists():
            shutil.copy(src_img, out_dir / "images" / split / src_img.name)
        else:
            skipped.append(f"{stem}: image file not found at {src_img}")

        lines = []
        for label, xmin, ymin, xmax, ymax in boxes:
            cls_id = class_to_id[label]
            cx = ((xmin + xmax) / 2) / width
            cy = ((ymin + ymax) / 2) / height
            bw = (xmax - xmin) / width
            bh = (ymax - ymin) / height
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        (out_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))

    # data.yaml
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    (out_dir / "data.yaml").write_text(f"""path: {out_dir.resolve()}
train: images/train
val: images/val

names:
{names_block}
""")

    if skipped:
        (out_dir / "skipped_boxes.log").write_text("\n".join(skipped))

    print(f"Processed {len(all_records)} images")
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train: {len(all_records) - len(val_records)} | Val: {len(val_records)}")
    if skipped:
        print(f"{len(skipped)} issues logged to {out_dir / 'skipped_boxes.log'}")
    print(f"Wrote {out_dir / 'data.yaml'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, help="Path to dataset-master/dataset-master")
    ap.add_argument("--out-dir", required=True, help="Output YOLO dataset folder")
    ap.add_argument("--val-split", type=float, default=0.2, help="Fraction of images held out for validation")
    args = ap.parse_args()
    build_dataset(args.dataset_dir, args.out_dir, args.val_split)


if __name__ == "__main__":
    main()