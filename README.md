# Nano-See

Traffic sign image classification. The dataset is prepared and ready to train;
no model has been trained yet.

## Layout

```
nano_see/
├── dataset/
│   ├── traffic_sign.csv         # class_id -> name (tracked in git)
│   ├── Images/<class_id>/       # raw source images, never modified
│   ├── clean/                   # generated: flat images + labels.csv
│   │   ├── images/
│   │   └── labels.csv
│   └── cls/                     # generated: Ultralytics layout
│       ├── train/<class>/
│       ├── val/<class>/
│       └── test/<class>/
├── pre_nano/                    # experiments
├── scripts/
│   ├── prepare_dataset.py       # raw -> clean
│   └── make_cls_layout.py       # clean -> cls
├── pyproject.toml
└── uv.lock
```

`dataset/Images`, `dataset/clean`, and `dataset/cls` are gitignored. Everything
except the raw images and `traffic_sign.csv` can be regenerated.

## Dataset

- 13,971 raw 32x32 PNGs in 58 class folders. Class 39 has no images and is
  absent, and some class names repeat in the CSV (e.g. four different "Level
  crossing countdown marker" signs), so the class ID is the real label.
- Images are a mix of 8-bit RGBA and 1-bit grayscale; they are copied
  byte-for-byte and never re-encoded.
- `clean/` holds 12,925 images after removing 304 duplicates whose content
  appears under more than one class (degenerate all-black / fully transparent
  frames left behind by augmentation) and 742 exact within-class duplicates.
- `clean/labels.csv` columns: `filename,class_id,class_name,split`.
- Split is stratified per class, 80/10/10, seed 42: 10,343 train / 1,291 val /
  1,291 test. Every class has at least 5 images in val and test.
- `cls/` mirrors the same split in Ultralytics classification layout. Folders
  are named `<class_id>_<slug>`, which keeps repeated names distinct and avoids
  `/` from names like `Maximum speed limit (90 km/h)`.

Caveat: random splitting is per image, so augmented siblings of the same source
sign can land in different splits. Val/test scores are therefore slightly
optimistic.

## Setup

```bash
uv sync
```

## Rebuild the datasets

```bash
uv run python scripts/prepare_dataset.py   # dataset/Images -> dataset/clean
uv run python scripts/make_cls_layout.py   # dataset/clean  -> dataset/cls
```

Both scripts delete and rewrite their output directory; the raw data is never
touched.

## Train

```bash
uv run yolo classify train data=dataset/cls model=yolo11n-cls.pt imgsz=32 epochs=50
```

Images are 32x32, so `imgsz=32` uses them at native resolution. The current
environment has no CUDA device (`torch.cuda.is_available() == False`), so
training runs on CPU. Results go to `runs/`, which is gitignored.
