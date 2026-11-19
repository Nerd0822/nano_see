from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"
DATASET_PATH = ROOT_DIR / "unified_dataset"
IMAGE_PATH = DATASET_PATH / "images"
LABEL_PATH = DATASET_PATH / "labels"
DATA_YAML_PATH = DATASET_PATH / "data.yaml"
BEST_PATH = RUNS_DIR / "signs" / "weights" / "best.pt"
MODEL = str(BASE_DIR / "yolo26n.pt") # 2.5m params

# ---- training ----
EPOCHS = 1
IMG_SIZE = 960  # CPU. Use 1280 if you train on a GPU
BATCH = 4  # CPU. Use 8-16 on GPU
SEED = 0
WORKERS = 4  # 6 physical cores; more just thrashes
DEVICE = "cpu"  # use 0 on a GPU
PATIENCE = 15  # early stop if no improvement for 15 epochs
CACHE = False  # 4K JPEGs would eat all 14 GB of RAM

# ---- augmentation (small dataset, tiny signs) ----
MOSAIC = 0.3  # mosaic shrinks signs; 1.0 is too aggressive here
SCALE = 0.5
