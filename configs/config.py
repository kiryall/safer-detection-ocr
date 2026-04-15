import os
from pathlib import Path

SEED = 42

"""Global configuration constants for the project."""

# Общие настройки
SEED = 42
SAVE_VISUALIZATION = False
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG"}
ALLOWLIST = "0123456789"  # Разрешённые символы для распознавания
DEBUG_SAVE_CROPS = False  # Сохранение кропов для отладки (влияет на производительность)
LOGGING_CONSOLE = False

# Путь к каталогам
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))

# Каталоги вывода
OUTPUT_CROPS_DIR = OUTPUT_DIR / "crops"
OUTPUT_VIS_DIR = OUTPUT_DIR / "visualizations"
OUTPUT_METADATA_DIR = OUTPUT_DIR / "metadata"
SUCCESS_DIR = OUTPUT_DIR / "success"
LOW_CONFIDENCE_DIR = OUTPUT_DIR / "low_confidence"
VERY_LOW_DIR = OUTPUT_DIR / "very_low"

# Каталоги данных
RAW_DATA_DIR = DATA_DIR / "raw"
DATASET_DIR = DATA_DIR / "dataset"
DATA_YAML = DATASET_DIR / "data.yaml"

# Пути к моделям
YOLO_MODEL_PATH = MODEL_DIR / "yolo"
OCR_MODEL_PATH = MODEL_DIR / "ocr"
YOLO_MODEL_NAME = os.getenv("YOLO_MODEL_NAME", "yolo26s-obb.pt")
YOLO_PRETRAINED_MODEL = Path(
    os.getenv("YOLO_PRETRAINED_MODEL", YOLO_MODEL_PATH / YOLO_MODEL_NAME)
)
MODEL_CONFIG_PATH = Path(os.getenv("MODEL_CONFIG_PATH", "configs"))
YOLO_BEST_MODEL = YOLO_MODEL_PATH / "v1" / "best.pt"
DIGIT_DETECTOR_MODEL = OCR_MODEL_PATH / "digit_detector.pt"

# Пороговые значения

CONF_THRESHOLD = 0.2 # порог уверенности детекции
OCR_ACCEPT_THRESHOLD = 0.8 # порог уверенности распознавания текста
CONFIDANCE_LEVEL_THRESHOLD = 0.8 # общий порог принятия результата

# Параметры OCR
OCR_THRESHOLD = 0.8
OCR_IOU_THRESHOLD = 0.8
OCR_MAX_DET = 5

# Параметры YOLO
YOLO_CONF_THRESHOLD = 0.2
YOLO_IOU_THRESHOLD = 0.2
