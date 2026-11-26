"""Create a reproducible toy microscopy dataset, without downloading any files."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from data import write_csv


def generate(root):
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(42)
    rows, prediction = [], []
    for index, split in enumerate(["train"] * 6 + ["val"] * 2 + ["test"] * 2):
        animal = f"demo-{index + 1:02d}"
        for field in range(2):
            # Odd dimensions exercise overlapping tiles and reconstruction.
            y, x = np.ogrid[:51, :59]
            mask = np.zeros((51, 59), dtype=bool)
            if index % 2 or field:
                for _ in range(3):
                    cy, cx = rng.integers(8, 43), rng.integers(8, 51)
                    mask |= ((x - cx) / 8) ** 2 + ((y - cy) / 5) ** 2 < 1
            raw = np.clip(rng.normal(220, 60, mask.shape) + mask * 2900, 0, 4095)
            stem = f"{animal}-{field + 1}"
            image_path, mask_path = stem + ".tif", stem + "-mask.png"
            Image.fromarray(raw.astype(np.uint16)).save(root / image_path)
            Image.fromarray(mask.astype(np.uint8) * 255).save(root / mask_path)
            rows.append(dict(animal_id=animal, split=split, image=image_path, mask=mask_path))
            if split == "test":
                prediction.append(dict(animal_id=animal, image=image_path))
    write_csv(root / "manifest.csv", rows)
    write_csv(root / "predict.csv", prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    generate(parser.parse_args().directory)

