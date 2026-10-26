"""
Faster R-CNN training pipeline — backbone-swappable, for WBC/BCCD-style detection.

Covers:
  1. Building the model with a chosen backbone + FPN
  2. Custom anchor generator tuned for small, roughly circular cells
  3. Dataset loading (COCO-format annotations, e.g. exported from BCCD)
  4. Training loop with LR scheduling and gradient clipping
  5. COCO-style mAP evaluation

Requires: torch, torchvision, pycocotools
    pip install torch torchvision pycocotools --break-system-packages
"""

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import misc as misc_nn_ops
from torch.utils.data import DataLoader
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# 1. BACKBONE + MODEL CONSTRUCTION
# ---------------------------------------------------------------------------

def build_model(backbone_name="resnet50", num_classes=5, pretrained=True):
    """
    num_classes = number of WBC subtypes + 1 (background)
    e.g. Neutrophil, Lymphocyte, Monocyte, Eosinophil, Basophil -> num_classes=6
    """

    if backbone_name in ["resnet50", "resnet101"]:
        # Easiest path: torchvision has a ready-made ResNet+FPN builder
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
            backbone_name, pretrained=pretrained, trainable_layers=3
        )
        # trainable_layers=3 means we fine-tune the top 3 ResNet stages,
        # freezing the earliest (most generic) filters — reduces overfitting
        # on small medical datasets.

    elif backbone_name == "densenet121":
        # DenseNet isn't natively supported by resnet_fpn_backbone, so we
        # build a custom FPN backbone manually.
        densenet = torchvision.models.densenet121(pretrained=pretrained)
        # DenseNet's feature blocks (denseblock1..4) give us 4 feature map
        # scales, same role as ResNet's layer1..4.
        return_layers = {
            "denseblock2": "0",
            "denseblock3": "1",
            "denseblock4": "2",
        }
        in_channels_list = [512, 1024, 1024]  # channel dims at those blocks
        backbone = torchvision.models.detection.backbone_utils.BackboneWithFPN(
            densenet.features, return_layers, in_channels_list, out_channels=256
        )

    elif backbone_name == "efficientnet_b0":
        effnet = torchvision.models.efficientnet_b0(pretrained=pretrained)
        return_layers = {"5": "0", "6": "1", "7": "2"}  # feature block indices
        in_channels_list = [112, 192, 320]
        backbone = torchvision.models.detection.backbone_utils.BackboneWithFPN(
            effnet.features, return_layers, in_channels_list, out_channels=256
        )

    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    # -----------------------------------------------------------------
    # 2. CUSTOM ANCHORS — tuned for small, round WBCs rather than COCO
    #    defaults (which assume cars/people-scale objects).
    #    WBCs in BCCD images are roughly 30-80px across at typical
    #    microscope zoom — so shrink anchor sizes and favor near-square
    #    aspect ratios since cells are round, not elongated.
    # -----------------------------------------------------------------
    anchor_generator = AnchorGenerator(
        sizes=((16, 32), (32, 64), (64, 128)),      # per FPN level
        aspect_ratios=((0.75, 1.0, 1.25),) * 3        # near-square only
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        # more proposals per image helps recall on dense/overlapping cells
        rpn_pre_nms_top_n_train=2000,
        rpn_post_nms_top_n_train=1000,
        box_score_thresh=0.05,
    )

    return model


# ---------------------------------------------------------------------------
# 3. DATASET (expects COCO-format json: images, annotations, categories)
# ---------------------------------------------------------------------------

class CocoDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_file, transforms=None):
        from pycocotools.coco import COCO
        self.img_folder = img_folder
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __getitem__(self, idx):
        import os
        from PIL import Image

        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]["file_name"]
        img = Image.open(os.path.join(self.img_folder, path)).convert("RGB")

        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)


def get_transform(train=True):
    transforms = [T.ToTensor()]
    if train:
        # Basic augmentation. See section on augmentation below for more.
        transforms.insert(0, T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1))
    return T.Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------------------------
# 4. TRAINING LOOP
# ---------------------------------------------------------------------------

def train(
    backbone_name="resnet50",
    num_classes=6,
    train_img_dir="data/train",
    train_ann_file="data/train_coco.json",
    val_img_dir="data/val",
    val_ann_file="data/val_coco.json",
    epochs=30,
    batch_size=4,
    lr=0.005,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model = build_model(backbone_name, num_classes).to(device)

    train_ds = CocoDetectionDataset(train_img_dir, train_ann_file, get_transform(True))
    val_ds = CocoDetectionDataset(val_img_dir, val_ann_file, get_transform(False))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=2
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    # Cosine schedule tends to outperform step decay for smaller datasets
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            # Gradient clipping stabilizes training, especially early on
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            running_loss += losses.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - lr: {scheduler.get_last_lr()[0]:.6f}")

        if (epoch + 1) % 5 == 0:
            evaluate(model, val_loader, device)
            torch.save(model.state_dict(), f"frcnn_{backbone_name}_epoch{epoch+1}.pth")

    return model


# ---------------------------------------------------------------------------
# 5. EVALUATION (COCO mAP)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data_loader, device):
    from pycocotools.cocoeval import COCOeval
    import numpy as np

    model.eval()
    coco_gt = data_loader.dataset.coco
    results = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for target, output in zip(targets, outputs):
            image_id = target["image_id"].item()
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                results.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                })

    if len(results) == 0:
        print("No predictions to evaluate.")
        return

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # Prints mAP@[.5:.95], mAP@.5, mAP@.75, and per-size AP (small/medium/large)
    # — the "small" AP number is the one to watch closely for WBC detection.


if __name__ == "__main__":
    # Example: train and compare each backbone in your study
    for backbone in ["resnet50", "densenet121", "efficientnet_b0"]:
        print(f"\n=== Training Faster R-CNN with {backbone} backbone ===")
        train(backbone_name=backbone, num_classes=6, epochs=30)