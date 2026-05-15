"""Moved-object detection with DETR on VIRAT frame pairs."""

import argparse
import json
import os
import random
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection


SEED = 42
SIZE = (800, 800)
TRAIN_FRAC = 0.8
NUM_CLASSES = 6
MODEL_NAME = "facebook/detr-resnet-50"

STRATEGIES = {
    "backbone":    ("model.backbone.",),
    "heads":       ("class_labels_classifier.", "bbox_predictor."),
    "transformer": ("model.encoder.", "model.decoder.", "model.query_position_embeddings"),
    "full":        ("",),   # empty prefix matches every param — no freezing
}
DEFAULT_LR = {"backbone": 1e-5, "heads": 1e-4, "transformer": 1e-4, "full": 1e-4}


##### PREPROCESSING #########

@dataclass
class Match:
    match_id: int
    box1: tuple
    box2: tuple
    obj_type: int


def parse_matches(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            rows.append((int(parts[0]), *map(float, parts[1:5]), int(parts[5])))
    if len(rows) % 2:
        raise ValueError(f"{path}: rows not paired")

    out = []
    for i in range(0, len(rows), 2):
        a, b = rows[i], rows[i + 1]
        if a[0] != b[0]:
            raise ValueError(f"{path}: id mismatch at row {i}")
        out.append(Match(a[0], a[1:5], b[1:5], a[5]))
    return out


def resize(img, size):
    H, W = size
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)


def scale_box(box, src, dst):
    sH, sW = src
    dH, dW = dst
    sx, sy = dW / sW, dH / sH
    x, y, w, h = box
    return x * sx, y * sy, w * sx, h * sy


def diff_pair(f1_path, f2_path, size):
    f1 = cv2.imread(f1_path)
    f2 = cv2.imread(f2_path)
    orig = f2.shape[:2]
    f1 = resize(f1, size)
    f2 = resize(f2, size)
    return cv2.absdiff(f2, f1), f2, orig


def load_sample(f1_path, f2_path, match_path, size=SIZE):
    matches = parse_matches(match_path)
    diff, f2, orig = diff_pair(f1_path, f2_path, size)
    boxes = [scale_box(m.box2, orig, size) for m in matches]
    labels = [m.obj_type for m in matches]
    return {"diff": diff, "f2": f2, "boxes": boxes, "labels": labels}


def resolve_pair(name, frames_root):
    name = os.path.basename(name)
    if name.endswith("_match.txt"):
        name = name[: -len("_match.txt")]
    folder, f1, f2 = name.split("-")
    return (os.path.join(frames_root, folder, f1 + ".png"),
            os.path.join(frames_root, folder, f2 + ".png"))


##### DATASET #########

class MovedObjects(Dataset):
    def __init__(self, files, match_dir, frames_root, size=SIZE):
        self.files = files
        self.match_dir = match_dir
        self.frames_root = frames_root
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        p1, p2 = resolve_pair(name, self.frames_root)
        s = load_sample(p1, p2, os.path.join(self.match_dir, name), self.size)
        diff_rgb = cv2.cvtColor(s["diff"], cv2.COLOR_BGR2RGB)
        anns = [{"bbox": list(box), "category_id": int(lab),
                 "area": float(box[2] * box[3]), "iscrowd": 0}
                for box, lab in zip(s["boxes"], s["labels"])]
        return {"image": diff_rgb, "target": {"image_id": idx, "annotations": anns}}


def make_collate(processor):
    def collate(batch):
        images  = [b["image"]  for b in batch]
        targets = [b["target"] for b in batch]
        enc = processor(images=images, annotations=targets, return_tensors="pt")
        return {"pixel_values": enc["pixel_values"],
                "pixel_mask":   enc["pixel_mask"],
                "labels":       enc["labels"]}
    return collate


def split_files(match_dir, frames_root):
    files = sorted(os.listdir(match_dir))

    def ok(n):
        p1, p2 = resolve_pair(n, frames_root)
        return os.path.isfile(p1) and os.path.isfile(p2)

    files = [f for f in files if ok(f)]
    random.Random(SEED).shuffle(files)
    cut = int(TRAIN_FRAC * len(files))
    return files[:cut], files[cut:]


##### FREEZE #########

def apply_freeze(model, strategy):
    prefixes = STRATEGIES[strategy]
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if any(name.startswith(pre) for pre in prefixes):
            p.requires_grad = True


def count_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


##### TRAINING #########

def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for i, batch in enumerate(loader):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask   = batch["pixel_mask"].to(device)
        labels       = [{k: v.to(device) for k, v in l.items()} for l in batch["labels"]]

        out = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        optimizer.zero_grad()
        out.loss.backward()
        optimizer.step()

        total += out.loss.item()
        if i % 20 == 0:
            print(f"  batch {i}/{len(loader)}  loss={out.loss.item():.4f}")
    return total / len(loader)


##### EVALUATION #########

def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (ua + 1e-9)


def cxcywh_to_xyxy(boxes, H, W):
    if len(boxes) == 0:
        return np.zeros((0, 4))
    out = []
    for cx, cy, w, h in boxes:
        cx, cy, w, h = cx * W, cy * H, w * W, h * H
        out.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return np.array(out)


@torch.no_grad()
def evaluate(model, loader, processor, device, score_thr=0.3, iou_thr=0.5):
    model.eval()
    tp = fp = fn = 0
    for batch in loader:
        out = model(pixel_values=batch["pixel_values"].to(device),
                    pixel_mask=batch["pixel_mask"].to(device))
        sizes = torch.tensor([[SIZE[0], SIZE[1]]] * len(batch["labels"]))
        preds = processor.post_process_object_detection(
            out, threshold=score_thr, target_sizes=sizes
        )
        for pred, gt in zip(preds, batch["labels"]):
            pb_arr = pred["boxes"].cpu().numpy()
            gb_arr = cxcywh_to_xyxy(gt["boxes"].cpu().numpy(), SIZE[0], SIZE[1])

            matched = set()
            for pb in pb_arr:
                best, j_best = 0.0, -1
                for j, gb in enumerate(gb_arr):
                    if j in matched:
                        continue
                    iou = box_iou(pb, gb)
                    if iou > best:
                        best, j_best = iou, j
                if best >= iou_thr:
                    tp += 1
                    matched.add(j_best)
                else:
                    fp += 1
            fn += len(gb_arr) - len(matched)

    p  = tp / (tp + fp + 1e-9)
    r  = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


##### PLOTS #########

def plot_curves(results, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    for strat, r in results.items():
        eps = range(1, len(r["loss"]) + 1)
        a1.plot(eps, r["loss"], marker="o", label=strat)
        a2.plot(eps, r["f1"],   marker="o", label=strat)
    a1.set_title("Training loss"); a1.set_xlabel("Epoch"); a1.set_ylabel("Loss")
    a2.set_title("Test F1");       a2.set_xlabel("Epoch"); a2.set_ylabel("F1")
    a2.set_ylim(0, 1)
    for ax in (a1, a2):
        ax.grid(True); ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def plot_predictions(model, test_files, match_dir, frames_root, processor, device, path, n=5):
    model.eval()
    rng = np.random.default_rng(0)
    chosen = rng.choice(len(test_files), size=n, replace=False)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))

    for i, idx in enumerate(chosen):
        name = test_files[idx]
        p1, p2 = resolve_pair(name, frames_root)
        s = load_sample(p1, p2, os.path.join(match_dir, name))
        diff_rgb = cv2.cvtColor(s["diff"], cv2.COLOR_BGR2RGB)
        f2_rgb   = cv2.cvtColor(s["f2"],   cv2.COLOR_BGR2RGB)

        inputs = processor(images=diff_rgb, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        sizes = torch.tensor([[SIZE[0], SIZE[1]]])
        pred = processor.post_process_object_detection(out, threshold=0.5, target_sizes=sizes)[0]

        axes[i, 0].imshow(diff_rgb); axes[i, 0].axis("off")
        axes[i, 0].set_title("I_diff + predictions")
        axes[i, 1].imshow(f2_rgb);   axes[i, 1].axis("off")
        axes[i, 1].set_title("Frame 2 — green=GT, red=pred")

        for box in pred["boxes"]:
            x1, y1, x2, y2 = box.cpu().numpy()
            for ax in axes[i]:
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                           fill=False, edgecolor="red", linewidth=2))
        for x, y, w, h in s["boxes"]:
            axes[i, 1].add_patch(plt.Rectangle((x, y), w, h, fill=False,
                                               edgecolor="lime", linewidth=2))

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


##### RUN #########

def build_model(device):
    return DetrForObjectDetection.from_pretrained(
        MODEL_NAME, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True
    ).to(device)


def run_strategy(strategy, train_dl, test_dl, processor, device,
                 epochs, warmup_epochs, lr, warmup_lr, save_dir):
    print(f"\n=== strategy: {strategy}  (warmup_lr={warmup_lr}, lr={lr})  ===")
    model = build_model(device)

    # Phase 1: warm-up — train everything, but with a smaller LR on the pretrained backbone
    # so its COCO weights don't get blown out (standard DETR fine-tuning recipe).
    if warmup_epochs > 0:
        print(f"\n-- warm-up: {warmup_epochs} epochs, all params trainable --")
        for p in model.parameters():
            p.requires_grad = True
        backbone_params = [p for n, p in model.named_parameters() if n.startswith("model.backbone.")]
        other_params    = [p for n, p in model.named_parameters() if not n.startswith("model.backbone.")]
        optim = AdamW(
            [{"params": backbone_params, "lr": warmup_lr * 0.1},
             {"params": other_params,    "lr": warmup_lr}]
        )
        for epoch in range(warmup_epochs):
            print(f"warmup epoch {epoch + 1}/{warmup_epochs}")
            loss = train_epoch(model, train_dl, optim, device)
            m = evaluate(model, test_dl, processor, device)
            print(f"  loss={loss:.4f}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")

    # Phase 2: strategy ablation — freeze per strategy, continue training the unfrozen block.
    print(f"\n-- ablation: {epochs} epochs, unfreezing only {strategy} --")
    apply_freeze(model, strategy)
    tr, tot = count_trainable(model)
    print(f"trainable: {tr:,} / {tot:,}  ({100 * tr / tot:.2f}%)")

    if strategy == "full":
        # Discriminative LR so the pretrained backbone is not blown out.
        backbone_params = [p for n, p in model.named_parameters() if n.startswith("model.backbone.")]
        other_params    = [p for n, p in model.named_parameters() if not n.startswith("model.backbone.")]
        optim = AdamW(
            [{"params": backbone_params, "lr": lr * 0.1},
             {"params": other_params,    "lr": lr}]
        )
    else:
        optim = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    history = {"loss": [], "precision": [], "recall": [], "f1": []}
    for epoch in range(epochs):
        print(f"epoch {epoch + 1}/{epochs}")
        loss = train_epoch(model, train_dl, optim, device)
        m = evaluate(model, test_dl, processor, device)
        history["loss"].append(loss)
        history["precision"].append(m["precision"])
        history["recall"].append(m["recall"])
        history["f1"].append(m["f1"])
        print(f"  loss={loss:.4f}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f"{strategy}.pt"))
    with open(os.path.join(save_dir, f"{strategy}.json"), "w") as f:
        json.dump(history, f, indent=2)
    return model, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["backbone", "heads", "transformer", "full", "all"], default="all")
    ap.add_argument("--epochs", type=int, default=15, help="strategy-ablation epochs (after warm-up)")
    ap.add_argument("--warmup-epochs", type=int, default=10, help="full-model warm-up epochs")
    ap.add_argument("--warmup-lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=None, help="overrides per-strategy default")
    ap.add_argument("--match-dir",  default="data/matched_annotations")
    ap.add_argument("--frames-root", default="../Assignment_3/data")
    ap.add_argument("--save-dir",   default="checkpoints")
    args = ap.parse_args()

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"device: {device}")

    processor = DetrImageProcessor.from_pretrained(MODEL_NAME)
    train_files, test_files = split_files(args.match_dir, args.frames_root)
    print(f"train: {len(train_files)}  test: {len(test_files)}")

    collate = make_collate(processor)
    train_ds = MovedObjects(train_files, args.match_dir, args.frames_root)
    test_ds  = MovedObjects(test_files,  args.match_dir, args.frames_root)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    strategies = ["backbone", "heads", "transformer", "full"] if args.strategy == "all" else [args.strategy]

    results = {}
    last_model = None
    for s in strategies:
        lr = args.lr if args.lr else DEFAULT_LR[s]
        last_model, results[s] = run_strategy(
            s, train_dl, test_dl, processor, device,
            args.epochs, args.warmup_epochs, lr, args.warmup_lr, args.save_dir,
        )

    # If only one strategy was run this time, fold in any prior runs' JSONs for the plot.
    for s in ["backbone", "heads", "transformer", "full"]:
        if s in results:
            continue
        path = os.path.join(args.save_dir, f"{s}.json")
        if os.path.isfile(path):
            with open(path) as f:
                results[s] = json.load(f)

    if results:
        plot_curves(results, os.path.join(args.save_dir, "loss_and_f1.png"))
        print("\nfinal metrics:")
        print(f"  {'strategy':<12}  {'P':>6}  {'R':>6}  {'F1':>6}")
        for s, r in results.items():
            print(f"  {s:<12}  {r['precision'][-1]:>6.3f}  {r['recall'][-1]:>6.3f}  {r['f1'][-1]:>6.3f}")
    if last_model is not None:
        plot_predictions(last_model, test_files, args.match_dir, args.frames_root,
                         processor, device, os.path.join(args.save_dir, "predictions.png"))


if __name__ == "__main__":
    main()
