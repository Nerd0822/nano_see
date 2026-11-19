from pathlib import Path

from config import (
    BATCH,
    BEST_PATH,
    CACHE,
    DATA_YAML_PATH,
    DEVICE,
    EPOCHS,
    IMG_SIZE,
    MODEL,
    MOSAIC,
    PATIENCE,
    RUNS_DIR,
    SCALE,
    SEED,
    WORKERS,
)
from ultralytics import YOLO


def main():
    data_yaml = str(DATA_YAML_PATH)
    model = YOLO(MODEL)
    model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        seed=SEED,
        workers=WORKERS,
        device=DEVICE,
        patience=PATIENCE,
        cache=CACHE,
        mosaic=MOSAIC,
        scale=SCALE,
        plots=True,
        exist_ok=True,
        project=str(RUNS_DIR),
        name="signs",
    )

    metrics = YOLO(str(BEST_PATH)).val(
        data=data_yaml,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        project=str(RUNS_DIR),
        name="val",
    )
    print("mAP50:", metrics.box.map50)


if __name__ == "__main__":
    main()
