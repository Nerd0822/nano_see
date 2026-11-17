#!/usr/bin/env python3
"""Build dataset/clean from the raw dataset.

- drops images whose exact content appears under more than one class
  (degenerate black/transparent frames produced by augmentation)
- drops exact within-class duplicates
- renames files to <class_id>_<seq>.png and flattens them into clean/images/
- writes clean/labels.csv with a stratified 80/10/10 train/val/test split

Raw data in dataset/Images and dataset/traffic_sign.csv is never modified.
"""
import csv
import hashlib
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "dataset"
DST = SRC / "clean"
SEED = 42


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    if not (SRC / "Images").is_dir():
        sys.exit(f"missing raw images at {SRC / 'Images'}")

    names = {
        int(row["ClassId"]): row["Name"]
        for row in csv.DictReader((SRC / "traffic_sign.csv").open(newline=""))
    }

    by_hash = defaultdict(list)
    classes = sorted(
        int(d.name) for d in (SRC / "Images").iterdir() if d.is_dir()
    )
    for c in classes:
        for path in sorted((SRC / "Images" / str(c)).iterdir()):
            if path.is_file():
                by_hash[md5(path)].append((c, path))

    cross = {h for h, v in by_hash.items() if len({c for c, _ in v}) > 1}
    print(f"dropping {len(cross)} cross-class hashes "
          f"({sum(len(by_hash[h]) for h in cross)} files)")

    kept = defaultdict(list)
    dupes = 0
    for h, v in by_hash.items():
        if h in cross:
            continue
        kept[v[0][0]].append(v[0][1])
        dupes += len(v) - 1
    print(f"dropping {dupes} exact within-class duplicates")

    if DST.exists():
        shutil.rmtree(DST)
    (DST / "images").mkdir(parents=True)

    rng = random.Random(SEED)
    rows = []
    for c in classes:
        paths = sorted(kept.get(c, []))
        if not paths:
            print(f"warning: class {c} has no images", file=sys.stderr)
            continue

        rng.shuffle(paths)
        n = len(paths)
        n_val = max(1, round(n * 0.1))
        n_test = max(1, round(n * 0.1))
        split = ["train"] * (n - n_val - n_test) + ["val"] * n_val + ["test"] * n_test
        assign = dict(zip(paths, split))

        for i, path in enumerate(sorted(paths), start=1):
            fn = f"{c:02d}_{i:04d}.png"
            shutil.copyfile(path, DST / "images" / fn)
            rows.append((fn, c, names.get(c, ""), assign[path]))

    with (DST / "labels.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "class_id", "class_name", "split"])
        w.writerows(rows)

    print(f"wrote {len(rows)} images and labels.csv to {DST}")

    # sanity checks
    files = list((DST / "images").iterdir())
    assert len(files) == len(rows)
    hashes = defaultdict(set)
    per_class = defaultdict(set)
    for fn, c, _, sp in rows:
        hashes[sp].add(md5(DST / "images" / fn))
        per_class[c].add(sp)
    assert not (hashes["train"] & hashes["val"])
    assert not (hashes["train"] & hashes["test"])
    assert not (hashes["val"] & hashes["test"])
    assert all(v == {"train", "val", "test"} for v in per_class.values())
    counts = {sp: len(h) for sp, h in hashes.items()}
    print("split:", counts)
    sizes = sorted(len(v) for v in kept.values())
    print(f"classes: {len(per_class)}, per-class images: min {sizes[0]}, max {sizes[-1]}")


if __name__ == "__main__":
    main()
