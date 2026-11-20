from config import BEST_PATH, DATASET_PATH, IMG_SIZE, RUNS_DIR
from ultralytics import YOLO

SOURCE = str(DATASET_PATH/"images"/"val")

model = YOLO(str(BEST_PATH))
model.predict(
    source=SOURCE,
    imgsz=IMG_SIZE,
    save=True,
    project=str(RUNS_DIR),
    name="predict",
    exist_ok=True,
)
print(f"SAVED to {RUNS_DIR / 'predict'}")
