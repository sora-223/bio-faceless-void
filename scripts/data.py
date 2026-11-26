"""Paired 2-D grayscale images, biological units and pixel counts."""
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Sample:
    animal: str
    split: str
    image: Path
    mask: Path | None


def plane(path):
    with Image.open(path) as image:
        if image.n_frames != 1 or image.mode not in {"L", "1", "I", "I;16", "I;16L", "I;16B", "F"}:
            raise ValueError(f"{path}: export a single grayscale plane and channel first")
        array = np.array(image)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{path}: expected finite 2-D pixels")
    return array


def mask_for(row, shape):
    mask = plane(row.mask)
    labels = np.unique(mask)
    foreground = labels[labels != 0]
    if mask.shape != shape or len(foreground) > 1 or not np.isin(foreground, [1, 255, 65535]).all():
        raise ValueError(f"{row.mask}: expected matching dimensions and binary 0/1, 0/255 or 0/65535")
    return mask > 0


def manifest(path, predict=False):
    path = Path(path).resolve()
    rows, seen, groups = [], set(), {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        keys = {"animal_id", "image"} if predict else {"animal_id", "split", "image", "mask"}
        if not keys.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: expected columns {sorted(keys)}")
        for i, record in enumerate(reader, 2):
            if any(not (record.get(key) or "").strip() for key in keys):
                raise ValueError(f"{path}, row {i}: missing required value")
            animal = record["animal_id"].strip()
            split = "predict" if predict else record["split"].strip()
            image = (path.parent / record["image"].strip()).resolve()
            mask = None if predict else (path.parent / record["mask"].strip()).resolve()
            if split not in {"predict", "train", "val", "test"} or (not predict and split == "predict"):
                raise ValueError(f"{path}, row {i}: invalid split")
            if groups.setdefault(animal, split) != split or image in seen:
                raise ValueError(f"{path}, row {i}: animal crosses splits or image repeated")
            if image == mask or not image.is_file() or (mask is not None and not mask.is_file()):
                raise ValueError(f"{path}, row {i}: missing file or image used as its own mask")
            seen.add(image)
            rows.append(Sample(animal, split, image, mask))
    if not rows:
        raise ValueError(f"{path}: empty manifest")
    return rows


def digest(array):
    h = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def inspect(rows):
    seen, hashes, pos = {}, {}, {}
    for row in rows:
        raw = plane(row.image)
        key = digest(raw)
        if key in seen and seen[key] != row.split:
            raise ValueError(f"{row.image}: identical pixels cross splits")
        seen[key] = row.split
        hashes[row.image] = key
        mask = mask_for(row, raw.shape)
        pos[row.split] = pos.get(row.split, 0) + int(mask.any())
    return hashes, pos


def scores(tp, fp, fn):
    denom = 2 * tp + fp + fn
    if denom == 0:
        return {"dice": None, "iou": None}
    return {"dice": 2 * tp / denom, "iou": tp / (tp + fp + fn)}


def counts(prediction, truth):
    return tuple(int(np.count_nonzero(x)) for x in
                 (prediction & truth, prediction & ~truth, ~prediction & truth))


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

