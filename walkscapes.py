from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT = Path(__file__).resolve().parent
WORK = PROJECT / "data"
LABELS = (
    "road", "sidewalk", "wall", "obstacle", "ground", "sky", "person",
    "vehicle", "vegetation", "bike path", "crosswalk", "animal", "train", "water",
)
COLORS = np.zeros((256, 3), dtype=np.uint8)
COLORS[:len(LABELS)] = [
    (128, 64, 128), (58, 220, 174), (102, 102, 156), (255, 139, 49),
    (145, 111, 68), (70, 130, 180), (240, 40, 70), (50, 75, 200),
    (107, 142, 35), (158, 95, 220), (255, 225, 90), (230, 90, 195),
    (0, 80, 100), (54, 171, 255),
]
SPLITS = ("train", "val", "test")
SIZE = (384, 640)


def sample_key(path: Path) -> str:
    return path.stem.lower().removesuffix("_mask")


def index_files(folder: Path, extensions: set[str]) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    result = {}
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() in extensions:
            key = sample_key(path)
            if key in result:
                raise ValueError(f"Duplicate sample {key}: {result[key]} and {path}")
            result[key] = path
    return result


def pairs_for(root: Path, split: str) -> list[tuple[Path, Path]]:
    imgs = index_files(root / "imgs" / split, {".png", ".jpg", ".jpeg"})
    masks = index_files(root / "masks" / split, {".png"})
    if not imgs or imgs.keys() != masks.keys():
        raise ValueError(f"{split}: {len(imgs)} images, {len(masks)} masks, "
                         f"{len(imgs.keys() & masks.keys())} pairs")
    return [(img, masks[key]) for key, img in imgs.items()]


def read_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image.resize(size, Image.Resampling.NEAREST))
    if mask.ndim != 2 or not np.all((mask < len(LABELS)) | (mask == 255)):
        raise ValueError(f"Expected indexed WalkScapes label IDs 0–13 or 255: {path}")
    return mask


def read_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize(size, Image.Resampling.BILINEAR)


def overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    blended = Image.blend(image, Image.fromarray(COLORS[mask]), 0.45)
    return Image.composite(blended, image, Image.fromarray((mask != 255).astype(np.uint8) * 255))


def preview(args: argparse.Namespace) -> None:
    pairs = pairs_for(args.data, args.split)
    rng = random.Random(42)
    args.output.mkdir(parents=True, exist_ok=True)
    for img_path, mask_path in rng.sample(pairs, min(args.count, len(pairs))):
        image = read_image(img_path, SIZE)
        mask = read_mask(mask_path, SIZE)
        overlay(image, mask).save(args.output / f"{img_path.stem}.png")
    print(f"{args.split}: {len(pairs)} pairs. Preview: {args.output}")


def preprocess(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.float32) / 255
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return ((rgb - mean) / std).transpose(2, 0, 1)


class Samples:
    def __init__(self, pairs: list[tuple[Path, Path]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image, mask = self.pairs[index]
        return preprocess(read_image(image, SIZE)), read_mask(mask, SIZE).astype(np.int64)


def metrics(model, loader, device) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    classes = len(LABELS)
    counts = torch.zeros(classes * classes, dtype=torch.int64, device=device)
    model.eval()
    with torch.inference_mode():
        for inputs, target in loader:
            target = target.to(device)
            logits = model(pixel_values=inputs.to(device)).logits
            prediction = F.interpolate(logits, target.shape[-2:], mode="bilinear",
                                       align_corners=False).argmax(dim=1)
            valid = target != 255
            counts += torch.bincount(
                target[valid] * classes + prediction[valid], minlength=classes * classes,
            )
    counts = counts.reshape(classes, classes).cpu().numpy()
    intersection = counts.diagonal()
    union = counts.sum(1) + counts.sum(0) - intersection
    iou = intersection[union > 0] / union[union > 0]
    scores = {LABELS[i]: float(intersection[i] / union[i]) for i in range(classes) if union[i]}
    scores["mIoU"] = float(iou.mean()) if iou.size else float("nan")
    return scores


def train(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import SegformerConfig, SegformerForSemanticSegmentation

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    pairs = {split: pairs_for(args.data, split) for split in SPLITS}
    print({split: len(items) for split, items in pairs.items()}, flush=True)
    config = SegformerConfig.from_pretrained(
        str(args.encoder), num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: index for index, label in enumerate(LABELS)},
        semantic_loss_ignore_index=255,
    )
    model = SegformerForSemanticSegmentation.from_pretrained(args.encoder, config=config).to(device)
    loaders = {
        split: DataLoader(Samples(items), batch_size=args.batch_size,
                          shuffle=split == "train", num_workers=args.workers,
                          persistent_workers=args.workers > 0, pin_memory=device.type == "cuda")
        for split, items in pairs.items()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, target in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = model(pixel_values=inputs.to(device), labels=target.to(device)).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        val = metrics(model, loaders["val"], device)
        mean_loss = running_loss / len(loaders["train"])
        history.append({"epoch": epoch, "loss": mean_loss, "validation": val})
        print(f"Epoch {epoch}: loss={mean_loss:.4f}, "
              f"validation mIoU={val['mIoU']:.4f}", flush=True)
        if val["mIoU"] > best:
            best = val["mIoU"]
            model.save_pretrained(args.output)
    model = SegformerForSemanticSegmentation.from_pretrained(args.output).to(device)
    test_scores = metrics(model, loaders["test"], device)
    report = {"samples": {key: len(value) for key, value in pairs.items()},
              "epochs": history, "test": test_scores}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Test IoU:", test_scores, flush=True)


def predict_mask(image: Image.Image, model, device, *, with_confidence=False,
                 with_surface_confidence=False, surface_ids=(0, 1)):
    import torch
    import torch.nn.functional as F

    inputs = torch.from_numpy(preprocess(image)).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(pixel_values=inputs).logits
        logits = F.interpolate(logits, (image.height, image.width), mode="bilinear",
                               align_corners=False)[0]
        if with_confidence or with_surface_confidence:
            probabilities = logits.softmax(dim=0)
            confidence, labels = probabilities.max(dim=0)
            if with_surface_confidence:
                surface = probabilities[list(surface_ids)].sum(dim=0)
                return labels.cpu().numpy(), confidence.cpu().numpy(), surface.cpu().numpy()
            return labels.cpu().numpy(), confidence.cpu().numpy()
        return logits.argmax(dim=0).cpu().numpy()


def predict(args: argparse.Namespace) -> None:
    import torch
    from transformers import SegformerForSemanticSegmentation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegformerForSemanticSegmentation.from_pretrained(args.model).to(device).eval()
    image = read_image(args.image, SIZE)
    mask = predict_mask(image, model, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    overlay(image, mask).save(args.output)
    if args.mask_output is not None:
        args.mask_output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8)).save(args.mask_output)
    print(f"Prediction: {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="WalkScapes semantic segmentation")
    commands = parser.add_subparsers(dest="command", required=True)
    data_default = WORK / "walkscapes-dataset-data" / "dataset"
    sample = commands.add_parser("preview")
    sample.add_argument("--data", type=Path, default=data_default)
    sample.add_argument("--split", choices=SPLITS, default="test")
    sample.add_argument("--count", type=int, default=4)
    sample.add_argument("--output", type=Path, default=PROJECT / "preview")
    sample.set_defaults(run=preview)
    training = commands.add_parser("train")
    training.add_argument("--data", type=Path, default=data_default)
    training.add_argument("--encoder", type=Path, default=WORK / "mit-b0")
    training.add_argument("--output", type=Path, default=PROJECT / "model")
    training.add_argument("--epochs", type=int, default=10)
    training.add_argument("--batch-size", type=int, default=4)
    training.add_argument("--workers", type=int, default=2)
    training.set_defaults(run=train)
    inference = commands.add_parser("predict")
    inference.add_argument("image", type=Path)
    inference.add_argument("--model", type=Path, default=PROJECT / "model")
    inference.add_argument("--output", type=Path, default=PROJECT / "prediction.png")
    inference.add_argument("--mask-output", type=Path, help="save raw 0–13 class IDs as a PNG")
    inference.set_defaults(run=predict)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
