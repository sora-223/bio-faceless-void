"""2-D binary U-Net. Run with --help for the four entry points."""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

from data import counts, inspect, manifest, mask_for, plane, scores, write_csv


def build_model():
    from torch import cat, nn

    def conv(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(),
                             nn.Conv2d(b, b, 3, padding=1), nn.ReLU())

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.ModuleList([conv(1, 16), conv(16, 32), conv(32, 64)])
            self.pool = nn.MaxPool2d(2)
            self.center = conv(64, 128)
            self.up = nn.ModuleList(nn.ConvTranspose2d(c * 2, c, 2, stride=2) for c in (64, 32, 16))
            self.decoder = nn.ModuleList(conv(c * 2, c) for c in (64, 32, 16))
            self.head = nn.Conv2d(16, 1, 1)

        def forward(self, x):
            skips = []
            for block in self.encoder:
                x = block(x)
                skips.append(x)
                x = self.pool(x)
            x = self.center(x)
            for up, block, skip in zip(self.up, self.decoder, reversed(skips)):
                x = block(cat((up(x), skip), dim=1))
            return self.head(x)

    return UNet()


def scaled(raw, scale):
    return np.clip(raw.astype(np.float32) / scale, 0, 1)


def padded(array, side):
    return np.pad(array, [(0, max(0, side - n)) for n in array.shape], mode="edge")


class Crops:
    def __init__(self, rows, side, scale, repeats, flip):
        self.rows, self.side, self.scale, self.repeats = rows, side, scale, repeats
        self.flip = flip

    def __len__(self):
        return len(self.rows) * self.repeats

    def __getitem__(self, index):
        import torch
        row = self.rows[index % len(self.rows)]
        raw = plane(row.image)
        mask = padded(mask_for(row, raw.shape), self.side)
        image = padded(scaled(raw, self.scale), self.side)
        y, x = (random.randrange(n - self.side + 1) for n in image.shape)
        region = np.s_[y:y+self.side, x:x+self.side]
        image, mask = image[region], mask[region]
        if self.flip and random.random() < 0.5:
            image, mask = np.fliplr(image), np.fliplr(mask)
        return (torch.from_numpy(image.copy())[None], torch.from_numpy(mask.astype(np.float32))[None])


def starts(length, side):
    last = length - side
    return sorted(set(range(0, last + 1, side // 2)) | {last})


def infer(model, raw, scale, side, device):
    import torch
    data = padded(scaled(raw, scale), side)
    total = np.zeros(data.shape, dtype=np.float32)
    hits = np.zeros(data.shape, dtype=np.uint8)
    model.eval()
    with torch.inference_mode():
        for y in starts(data.shape[0], side):
            for x in starts(data.shape[1], side):
                area = np.s_[y:y+side, x:x+side]
                tensor = torch.from_numpy(data[area].copy())[None, None].to(device)
                total[area] += model(tensor).sigmoid()[0, 0].cpu().numpy()
                hits[area] += 1
    h, w = raw.shape
    return (total[:h, :w] / hits[:h, :w]) >= 0.5


def device_for(value):
    import torch
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable. Check the NVIDIA driver and PyTorch build or select cpu.")
    return ("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else value


def new_output(path):
    path.mkdir(parents=True, exist_ok=False)


def train(args, rows):
    import torch
    from torch.utils.data import DataLoader
    dev = [r for r in rows if r.split in {"train", "val"}]
    hashes, pos = inspect(dev)
    if not pos.get("train") or not pos.get("val"):
        raise ValueError("Training and validation each need at least one foreground annotation.")
    if not math.isfinite(args.scale) or args.scale <= 0 or args.tile < 16 or args.tile % 8:
        raise ValueError("scale must be positive and finite; tile must be >=16 and divisible by 8")
    if min(args.epochs, args.batch, args.crops) < 1:
        raise ValueError("epochs, batch and crops must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = device_for(args.device)
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loader = DataLoader(Crops([r for r in dev if r.split == "train"], args.tile, args.scale, args.crops, args.flip),
                        batch_size=args.batch, shuffle=True)
    val = [r for r in dev if r.split == "val"]
    new_output(args.out)
    settings = {k: getattr(args, k) for k in ("tile", "scale", "epochs", "batch", "seed", "crops", "flip")}
    settings.update(device=device, torch_version=str(torch.__version__), threshold=0.5)
    (args.out / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    best = -1
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(x)
        totals = np.zeros(3, dtype=np.int64)
        for row in val:
            raw = plane(row.image)
            pred = infer(model, raw, args.scale, args.tile, device)
            totals += counts(pred, mask_for(row, raw.shape))
        score = scores(*map(int, totals))["dice"]
        history.append({"epoch": epoch, "loss": loss_sum / len(loader.dataset), "val_dice": score})
        write_csv(args.out / "history.csv", history)
        if score > best:
            best = score
            state = {"model": model.state_dict(), "epoch": epoch, "settings": settings,
                     "dev_animals": sorted({r.animal for r in dev}), "dev_hashes": sorted(set(hashes.values()))}
            temporary = args.out / "best.pt.tmp"
            torch.save(state, temporary)
            temporary.replace(args.out / "best.pt")
        print(f"Epoch {epoch}: validation Dice {score:.4f}", flush=True)


def export(args, rows, evaluate):
    import torch
    device = device_for(args.device)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    cfg = state["settings"]
    model = build_model()
    model.load_state_dict(state["model"])
    model.to(device)
    if evaluate:
        rows = [r for r in rows if r.split == "test"]
        if not rows:
            raise ValueError("No held-out test rows")
        hashes, _ = inspect(rows)
        if set(state["dev_animals"]) & {r.animal for r in rows} or set(state["dev_hashes"]) & set(hashes.values()):
            raise ValueError("Test data overlaps the model's training or validation data")
    new_output(args.out)
    images, animals = [], {}
    for i, row in enumerate(rows, 1):
        raw = plane(row.image)
        pred = infer(model, raw, cfg["scale"], cfg["tile"], device)
        base = f"{i:05d}"
        Image.fromarray(pred.astype(np.uint8) * 255).save(args.out / (base + "_mask.png"))
        gray = (scaled(raw, cfg["scale"]) * 255).astype(np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        rgb[pred, 0] = (rgb[pred, 0].astype(np.uint16) + 255) // 2
        rgb[pred, 1:] //= 2
        Image.fromarray(rgb).save(args.out / (base + "_overlay.png"))
        pixels = raw.size
        fg = int(pred.sum())
        entry = {"animal_id": row.animal, "image": str(row.image), "output_id": base,
                 "image_pixels": pixels, "positive_pixels": fg, "area_fraction": fg / pixels,
                 "positive_mean_raw": float(raw[pred].mean()) if fg else None}
        group = animals.setdefault(row.animal, dict(images=0, pixels=0, positive=0, raw_sum=0., tp=0, fp=0, fn=0))
        group["images"] += 1
        group["pixels"] += pixels
        group["positive"] += fg
        group["raw_sum"] += raw[pred].sum(dtype=np.float64)
        if evaluate:
            tp, fp, fn = counts(pred, mask_for(row, raw.shape))
            entry.update(tp=tp, fp=fp, fn=fn, **scores(tp, fp, fn))
            for k in ("tp", "fp", "fn"):
                group[k] += entry[k]
        images.append(entry)
    write_csv(args.out / "images.csv", images)
    summary = []
    for animal, g in animals.items():
        entry = dict(animal_id=animal, images=g["images"], total_pixels=g["pixels"],
                     positive_pixels=g["positive"], area_fraction=g["positive"] / g["pixels"],
                     positive_mean_raw=g["raw_sum"] / g["positive"] if g["positive"] else None)
        if evaluate:
            entry.update(tp=g["tp"], fp=g["fp"], fn=g["fn"], **scores(g["tp"], g["fp"], g["fn"]))
        summary.append(entry)
    write_csv(args.out / "animals.csv", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "train", "evaluate", "predict"):
        p = sub.add_parser(name)
        p.add_argument("manifest", type=Path)
        if name != "audit":
            p.add_argument("--out", type=Path, required=True, help="new output directory")
            p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
        if name in ("evaluate", "predict"):
            p.add_argument("--weights", required=True, type=Path)
        if name == "train":
            p.add_argument("--scale", required=True, type=float, help="raw intensity mapped to 1")
            p.add_argument("--tile", type=int, default=256)
            p.add_argument("--epochs", type=int, default=20)
            p.add_argument("--batch", type=int, default=2)
            p.add_argument("--crops", type=int, default=4, help="random crops per image per epoch")
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--flip", action="store_true", help="allow paired left/right flips when orientation is exchangeable")
    args = parser.parse_args()
    rows = manifest(args.manifest, predict=args.command == "predict")
    if args.command == "audit":
        _, pos = inspect(rows)
        for split in sorted(pos):
            data = [r for r in rows if r.split == split]
            print(f"{split}: {len(set(r.animal for r in data))} animals, {len(data)} images, "
                  f"{pos[split]} positive, {len(data)-pos[split]} empty masks")
    elif args.command == "train":
        train(args, rows)
    else:
        export(args, rows, evaluate=args.command == "evaluate")


if __name__ == "__main__":
    main()

