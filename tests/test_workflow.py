"""Exercise the published CLI on generated data and verify its measurements."""
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from data import Sample, counts, inspect, manifest, mask_for, plane, scores, write_csv
from demo import generate


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class Workflow(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.data = self.root / "data"
        generate(self.data)
        self.csv = self.data / "manifest.csv"

    def cli(self, *args, ok=True):
        result = subprocess.run([sys.executable, str(SCRIPTS / "unet.py"), *map(str, args)],
                                capture_output=True, text=True,
                                env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
        if ok:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def test_labels_and_metrics(self):
        self.assertEqual(scores(1, 1, 1), {"dice": 0.5, "iou": 1 / 3})
        self.assertIsNone(scores(0, 0, 0)["dice"])
        row = manifest(self.csv)[0]
        self.assertFalse(mask_for(row, plane(row.image).shape).any())
        labels = self.root / "labels.tif"
        sample = Sample("a", "train", row.image, labels)
        for values in ([0, 65535], [1, 255]):
            Image.fromarray(np.array([values], dtype=np.uint16)).save(labels)
            if values[0] == 0:
                self.assertEqual(mask_for(sample, (1, 2)).tolist(), [[False, True]])
            else:
                with self.assertRaises(ValueError):
                    mask_for(sample, (1, 2))

    def test_split_leakage(self):
        rows = read_csv(self.csv)
        rows[-1]["animal_id"] = rows[0]["animal_id"]
        bad = self.data / "bad.csv"
        write_csv(bad, rows)
        with self.assertRaisesRegex(ValueError, "animal crosses"):
            manifest(bad)
        rows = manifest(self.csv)
        duplicate = self.root / "duplicate.tif"
        shutil.copyfile(rows[0].image, duplicate)
        with self.assertRaisesRegex(ValueError, "identical pixels"):
            inspect([rows[0], Sample("other", "test", duplicate, rows[0].mask)])

    def test_end_to_end(self):
        audit = self.cli("audit", self.csv)
        self.assertIn("2 animals, 4 images", audit.stdout)
        run = self.root / "run"
        self.cli("train", self.csv, "--out", run, "--scale", "4095", "--device", "cpu",
                 "--epochs", "3", "--tile", "32", "--batch", "2", "--crops", "3")
        self.assertEqual(len(read_csv(run / "history.csv")), 3)
        weights = run / "best.pt"
        results, new = self.root / "eval", self.root / "new"
        self.cli("evaluate", self.csv, "--weights", weights, "--out", results, "--device", "cpu")
        self.cli("predict", self.data / "predict.csv", "--weights", weights, "--out", new, "--device", "cpu")
        slices = read_csv(results / "images.csv")
        self.assertEqual(len(slices), 4)
        truths = {str(r.image): r for r in manifest(self.csv) if r.split == "test"}
        for record in slices:
            r = truths[record["image"]]
            name = record["output_id"]
            pred = plane(results / (name + "_mask.png")) > 0
            np.testing.assert_array_equal(pred, plane(new / (name + "_mask.png")) > 0)
            raw = plane(r.image)
            self.assertEqual(pred.shape, raw.shape)
            with Image.open(new / (name + "_overlay.png")) as overlay:
                self.assertEqual(overlay.size, (raw.shape[1], raw.shape[0]))
            tp, fp, fn = counts(pred, mask_for(r, raw.shape))
            self.assertEqual((tp, fp, fn), tuple(int(record[k]) for k in ("tp", "fp", "fn")))
            self.assertAlmostEqual(float(record["area_fraction"]), pred.mean())
            if pred.any():
                self.assertAlmostEqual(float(record["positive_mean_raw"]), raw[pred].mean())
        animals = read_csv(results / "animals.csv")
        self.assertEqual(len(animals), 2)
        for group in animals:
            fields = [r for r in slices if r["animal_id"] == group["animal_id"]]
            fraction = sum(int(r["positive_pixels"]) for r in fields) / sum(int(r["image_pixels"]) for r in fields)
            self.assertAlmostEqual(float(group["area_fraction"]), fraction)
        records = read_csv(self.csv)
        leaked = [dict(records[1], animal_id="renamed", split="test")]
        bad = self.data / "leaked.csv"
        write_csv(bad, leaked)
        reject = self.cli("evaluate", bad, "--weights", weights, "--out", self.root / "invalid",
                          "--device", "cpu", ok=False)
        self.assertIn("overlaps", reject.stderr)
        self.assertFalse((self.root / "invalid").exists())


if __name__ == "__main__":
    unittest.main()

