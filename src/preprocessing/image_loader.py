# src/preprocessing/image_loader.py

from pathlib import Path
from configs.config import RAW_DATA_DIR


def load_image(image_path: Path = RAW_DATA_DIR):
    """Загружает изображение по указанному пути.

    Args:
        image_path: Путь к изображению.

    Returns:
        Загруженное изображение.
    """
    # Реализация загрузки изображения (например, с помощью OpenCV или PIL)
    pass