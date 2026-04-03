import os
from pathlib import Path

# Общие директории
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
OUTPUT_CROPS_DIR = OUTPUT_DIR / "crops"
OUTPUT_VIS_DIR = OUTPUT_DIR / "visualizations"
OUTPUT_METADATA_DIR = OUTPUT_DIR / "metadata"

# Директории данных
RAW_DATA_DIR = DATA_DIR / "raw"

# Директории для обучения
DATASET_DIR = DATA_DIR / "dataset"
DATA_YAML = DATASET_DIR / "data.yaml"

# Пути к моделям
YOLO_MODEL_PATH = MODEL_DIR / "yolo"
OCR_MODEL_PATH = MODEL_DIR / "ocr"
YOLO_MODEL_NAME = os.getenv("YOLO_MODEL_NAME", "yolov8n.pt")
YOLO_PRETRAINED_MODEL = Path(
    os.getenv("YOLO_PRETRAINED_MODEL", YOLO_MODEL_PATH / YOLO_MODEL_NAME)
)
MODEL_CONFIG_PATH = Path(os.getenv("MODEL_CONFIG_PATH", "configs"))

YOLO_BEST_MODEL = YOLO_MODEL_PATH / "v1" / "best.pt"

CONF_THRESHOLD = 0.5

SAVE_VISUALIZATION = True

# параметры yolo
YOLO_CONF_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.5

SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG"}