# src\preprocessing\image_utils.py

from pathlib import Path
from typing import Optional, List
from PIL import Image
import cv2
import numpy as np

from configs.config import CONF_THRESHOLD, OUTPUT_CROPS_DIR
from src.utils.logger import setup_logging

logger = setup_logging(
    log_file="image_cropper.log",
    console=True,
    remove_file=True,
    logger_name="image_cropper",
)


class ImageCropper:
    """
    Отдельный класс для обрезки изображений по bbox и сохранения кропов.
    """

    def __init__(self, save_crops: bool = True, crops_base_dir: str = OUTPUT_CROPS_DIR):
        self.save_crops = save_crops
        self.crops_base_dir = Path(crops_base_dir)

        if save_crops:
            self.crops_base_dir.mkdir(parents=True, exist_ok=True)
            for subdir in ["success", "low_confidence", "no_tablet"]:
                (self.crops_base_dir / subdir).mkdir(exist_ok=True)

    def crop_and_save(
        self, image_path: Path, bbox: List[int], confidence: float = 0.0
    ) -> tuple[Optional[Image.Image], Optional[Path]]:
        """
        Обрезает изображение и optionally сохраняет кроп.
        Args:
            image_path: Путь к исходному изображению.
            bbox: Список координат [x1, y1, x2, y2] для обрезки.
            confidence: Уровень confidence для определения папки сохранения.
        Returns:
            Кортеж (обрезанное изображение в формате PIL, путь к сохраненному кропу) или (None, None) если обрезка не удалась.
        """
        try:
            logger.info(
                f"Обрезка изображения: {image_path} с bbox: {bbox} и confidence: {confidence:.3f}"
            )

            img_pil = Image.open(image_path).convert("RGB")
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            x1, y1, x2, y2 = bbox
            cropped_cv = img_cv[y1:y2, x1:x2]

            if cropped_cv.size == 0:
                return (
                    img_pil,
                    None,
                )  # Возвращаем оригинальное изображение, если кроп пустой

            cropped_pil = Image.fromarray(cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2RGB))

            crop_path = None
            if self.save_crops:
                crop_path = self._get_crop_path(image_path.stem, confidence)
                self._save_crop(cropped_pil, crop_path)

            return cropped_pil, crop_path

        except Exception as e:
            logger.error(f"Ошибка при обрезке изображения {image_path}: {e}")
            return None, None

    def _get_crop_path(self, stem: str, confidence: float) -> Path:
        """Определяет путь для сохранения кропа."""
        # Определяем папку для сохранения в зависимости от confidence
        if confidence >= CONF_THRESHOLD:
            subdir = "success"
            prefix = ""
        else:
            subdir = "low_confidence" if confidence > 0 else "no_tablet"
            prefix = "!" if confidence > 0 else "no_tablet"

        filename = f"{prefix}{stem}.jpg"
        return self.crops_base_dir / subdir / filename

    def _save_crop(self, cropped_pil: Image.Image, save_path: Path):
        """Сохраняет кроп в указанный путь."""
        logger.info(f"Сохранение кропа: {save_path}")
        cropped_pil.save(save_path, quality=95)
