#!/usr/bin/env python3
"""Build dataset/cls (Ultralytics classification layout) from dataset/clean.

Ultralytics expects <root>/<split>/<class_name>/*.png. Class folders are named
<class_id>_<slug> so repeated names in traffic_sign.csv stay unambiguous and
characters like "/" in "90 km/h" never reach a path.
"""
import csv
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "dataset" / "clean"
DST = ROOT / "dataset" / "cls"
SPLITS = ("train", "val", "test")


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "class"


def main():
    labels = SRC / "labels.csv"
    if not labels.is_file():
        sys.exit(f"missing {labels}; run scripts/prepare_dataset.py first")

    rows = list(csv.DictReader(labels.open(newline="")))

    folder = {}
    used = set()
    for r in rows:
        cid = int(r["class_id"])
        if cid in folder:
            continue
        base = f"{cid:02d}_{slug(r['class_name'])}"
        name, n = base, 2
        while name in used:
            name, n = f"{base}_{n}", n + 1
        used.add(name)
        folder[cid] = name

    if DST.exists():
        shutil.rmtree(DST)
    for sp in SPLITS:
        for name in folder.values():
            (DST / sp / name).mkdir(parents=True)

    for r in rows:
        shutil.copyfile(
            SRC / "images" / r["filename"],
            DST / r["split"] / folder[int(r["class_id"])] / r["filename"],
        )

    counts = Counter()
    for sp in SPLITS:
        for d in (DST / sp).iterdir():
            counts[(sp, d.name)] = len(list(d.iterdir()))

    for sp in SPLITS:
        n = sum(v for (s, _), v in counts.items() if s == sp)
        print(f"{sp:5}: {n:5d} images, "
              f"{len({name for (s, name) in counts if s == sp})} classes")
    print(f"total: {sum(counts.values())} images, {len(folder)} classes -> {DST}")
    assert sum(counts.values()) == len(rows)
    assert all(v > 0 for v in counts.values())

    repeated = {}
    for r in rows:
        repeated.setdefault(r["class_name"], set()).add(int(r["class_id"]))
    for name, ids in repeated.items():
        if len(ids) > 1:
            print(f"  repeated name {name!r}: {[folder[i] for i in sorted(ids)]}")


if __name__ == "__main__":
    main()
