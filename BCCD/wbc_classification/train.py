"""
Train a YOLO classifier on the WBC subtype dataset (dataset2-master).

Dataset layout (see ../wbc_classification/dataset), built as junctions
pointing into dataset2-master so no images are duplicated on disk:
    dataset/train/{EOSINOPHIL,LYMPHOCYTE,MONOCYTE,NEUTROPHIL}/*.jpeg
    dataset/val/{EOSINOPHIL,LYMPHOCYTE,MONOCYTE,NEUTROPHIL}/*.jpeg
    dataset/test/{...}   (small hand-picked TEST_SIMPLE set)

Usage:
    python train.py --epochs 30 --imgsz 224 --model yolo11n-cls.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

DATASET_DIR = Path(__file__).parent / "dataset"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n-cls.pt", help="Pretrained classification checkpoint to start from")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(DATASET_DIR),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(Path(__file__).parent / "runs"),
        name="wbc_cls",
    )

    metrics = model.val(
        data=str(DATASET_DIR),
        split="test",
        project=str(Path(__file__).parent / "runs"),
        name="wbc_cls_test",
    )
    print(metrics)


if __name__ == "__main__":
    main()
