"""Конфигурация UI приложения (без дублирования configs/config.py)."""

import sys
from pathlib import Path

# Добавляем корень проекта в PATH ДО импорта configs
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Импортируем все из главного конфига
from configs.config import (  # noqa: E402
    CONF_THRESHOLD,
    CONFIDANCE_LEVEL_THRESHOLD,
    DATA_DIR,
    DEBUG_SAVE_CROPS,
    LOG_DIR,
    LOW_CONFIDENCE_DIR,
    MODEL_DIR,
    OCR_ACCEPT_THRESHOLD,
    OUTPUT_DIR,
    OUTPUT_CROPS_DIR,
    SUCCESS_DIR,
    VERY_LOW_DIR,
    YOLO_CONF_THRESHOLD,
)

# Вычисляем PROJECT_ROOT на основе структуры
PROJECT_ROOT = (
    OUTPUT_DIR.parent.parent
    if OUTPUT_DIR.parent.parent.name == "safer"
    else Path(__file__).resolve().parent.parent.parent
)

# UI-специфичные константы (не дублируем из configs/config.py)
APP_TITLE = "SEFER Vision — Обработка фотографий табличек"
PAGE_LAYOUT = "wide"
SIDEBAR_STATE = "expanded"

# Пути к отчётам
REPORT_CSV = OUTPUT_DIR / "sefer_report.csv"
REPORT_TXT = OUTPUT_DIR / "sefer_summary.txt"

# Экспортируем все используемые в UI константы
__all__ = [
    "CONF_THRESHOLD",
    "CONFIDANCE_LEVEL_THRESHOLD",
    "DATA_DIR",
    "DEBUG_SAVE_CROPS",
    "LOG_DIR",
    "LOW_CONFIDENCE_DIR",
    "MODEL_DIR",
    "OCR_ACCEPT_THRESHOLD",
    "OUTPUT_DIR",
    "OUTPUT_CROPS_DIR",
    "PROJECT_ROOT",
    "REPORT_CSV",
    "REPORT_TXT",
    "SUCCESS_DIR",
    "VERY_LOW_DIR",
    "YOLO_CONF_THRESHOLD",
    "APP_TITLE",
    "PAGE_LAYOUT",
    "SIDEBAR_STATE",
]
