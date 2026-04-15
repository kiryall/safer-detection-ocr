# src\preprocessing\image_utils.py

import math
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from PIL import Image
import cv2
import numpy as np

from configs.config import (
    CONF_THRESHOLD,
    OUTPUT_CROPS_DIR,
    LOGGING_CONSOLE,
)
from src.utils.logger import setup_logging

logger = setup_logging(
    log_file="image_cropper.log",
    console=LOGGING_CONSOLE,
    remove_file=True,
    logger_name="image_cropper",
    level="DEBUG",
)


class ImageCropper:
    """
    Обрезка изображений:
    - По bbox (старый способ)
    - По полигону 4 точки + перспективная трансформация + выравнивание
    """

    def __init__(self,
                 save_crops: bool = True,
                 crops_base_dir: str = OUTPUT_CROPS_DIR,
                 conf_threshold: float = None):
        self.save_crops = save_crops
        self.crops_base_dir = Path(crops_base_dir)
        self.conf_threshold = conf_threshold or CONF_THRESHOLD

        if self.save_crops:
            self.crops_base_dir.mkdir(parents=True, exist_ok=True)
            for subdir in ["success"]:
                (self.crops_base_dir / subdir).mkdir(exist_ok=True)

    def crop_obb(
        self,
        image_path: Path,
        obb: List[float],                    # [cx, cy, w, h, angle_rad]
        confidence: float = 0.0,
        debug: bool = False
    ) -> Tuple[Optional[Image.Image], Optional[Path], Dict]:
        """
        Обрезка и выравнивание OBB в формате xywhr.
        Args:
            image_path: Путь к исходному изображению.
            obb: Список из 5 значений [center_x, center_y, width, height, angle_radians].
            confidence: Уровень confidence детекции для определения папки сохранения.
            debug: Если True, сохраняет промежуточные изображения для отладки.
        Returns:
            Кортеж (обрезанное и выровненное изображение в формате PIL, путь к сохраненному кропу) или (None, None) если обрезка не удалась.
        """
        try:
            cx, cy, w, h, angle_rad = obb[:5]
            angle_deg = math.degrees(angle_rad)

            logger.info(f"OBB-кроп: {image_path.name} | conf={confidence:.3f} | angle={angle_deg:.1f}°")

            img_cv = cv2.imread(str(image_path))
            if img_cv is None:
                logger.error(f"Не удалось загрузить {image_path}")
                return None, None, {}

            orig_h, orig_w = img_cv.shape[:2]

            # === 1. Добавляем запас (margin) — очень важно! ===
            margin = int(max(w, h) * 0.15)   # 15% запас по размеру
            x1 = max(0, int(cx - w/2 - margin))
            y1 = max(0, int(cy - h/2 - margin))
            x2 = min(orig_w, int(cx + w/2 + margin))
            y2 = min(orig_h, int(cy + h/2 + margin))

            # Вырезаем область с запасом
            crop_rect = img_cv[y1:y2, x1:x2]
            if crop_rect.size == 0:
                logger.warning(f"Пустой прямоугольник для {image_path.name}")
                cropped_pil = Image.open(image_path).convert("RGB")
                enhanced_pil = self._enhance_for_ocr_pil(cropped_pil)
                return enhanced_pil, None, {}

            # Новые координаты центра относительно вырезанного куска
            new_cx = cx - x1
            new_cy = cy - y1

            # === 2. Поворачиваем только вырезанный кусок ===
            rotation_matrix = cv2.getRotationMatrix2D((new_cx, new_cy), angle_deg, 1.0)

            rotated = cv2.warpAffine(
                crop_rect,
                rotation_matrix,
                (x2 - x1, y2 - y1),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

            # === 3. Обрезаем уже повёрнутое изображение по размеру таблички ===
            rot_h, rot_w = rotated.shape[:2]
            half_w = int(w / 2)
            half_h = int(h / 2)

            # Центр после поворота (примерно)
            rx1 = max(0, int(new_cx - half_w))
            ry1 = max(0, int(new_cy - half_h))
            rx2 = min(rot_w, int(new_cx + half_w))
            ry2 = min(rot_h, int(new_cy + half_h))

            final_crop = rotated[ry1:ry2, rx1:rx2]

            if final_crop.size == 0 or final_crop.shape[0] < 30 or final_crop.shape[1] < 30:
                logger.warning(f"Финальный кроп слишком маленький для {image_path.name}")
                final_pil = Image.fromarray(cv2.cvtColor(crop_rect, cv2.COLOR_BGR2RGB))
            else:
                final_pil = Image.fromarray(cv2.cvtColor(final_crop, cv2.COLOR_BGR2RGB))

            # Улучшение для OCR
            enhanced_pil = self._enhance_for_ocr_pil(final_pil)

            # Сохранение для дебага
            crop_path = None
            if self.save_crops:
                crop_path = self._save_crop(enhanced_pil, image_path.stem, confidence)

            metadata = {
                "center_x": cx, "center_y": cy,
                "width": w, "height": h,
                "angle_deg": angle_deg,
                "final_width": final_crop.shape[1],
                "final_height": final_crop.shape[0],
            }

            return enhanced_pil, crop_path, metadata

        except Exception as e:
            logger.error(f"Ошибка OBB-кропа {image_path.name}: {e}")
            return None, None, {}

    def save_original(self, image_path: Path, confidence: float) -> Path:
        """Сохраняет оригинальное изображение в соответствующую папку. Возвращает путь сохранения."""
        if confidence >= self.conf_threshold:
            subdir = "success"
        elif confidence < self.conf_threshold and confidence != 0 and not None:
            subdir = "low_confidence"
        else:
            subdir = "very_low"

        filename = f"{image_path.stem}.jpg"
        save_path = self.crops_base_dir / subdir / filename

        # Копируем оригинальное изображение
        img_pil = Image.open(image_path).convert("RGB")
        img_pil.save(save_path, quality=92)
        logger.debug(f"Оригинал сохранён → {save_path}")
        return save_path

    def _save_crop(
        self, cropped_pil: Image.Image, stem: str, confidence: float
    ) -> Path:
        """Сохраняет кроп"""
        logger.debug(f"_save_crop confidence = {confidence}")
        logger.debug(f"CONF_THRESHOLD = {self.conf_threshold}")

        if confidence >= self.conf_threshold:
            subdir = "success"
        else:
            return None

        filename = f"{stem}.jpg"
        save_path = self.crops_base_dir / subdir / filename

        cropped_pil.save(save_path, quality=92)
        logger.debug(f"Кроп сохранён → {save_path}")
        return save_path

    def _enhance_for_ocr_pil(self, pil_image: Image.Image) -> Image.Image:
        """Улучшение изображения для OCR."""
        img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        return Image.fromarray(sharpened, mode="L")


def resize_image_if_large(image_path: Path, max_dimension: int = 640) -> np.ndarray:
    """
    Загружает изображение и уменьшает его, если максимальная сторона превышает max_dimension.
    Возвращает numpy array в формате RGB.

    Args:
        image_path: Путь к изображению.
        max_dimension: Максимальный размер стороны (по умолчанию 640).

    Returns:
        Numpy array изображения в формате RGB.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    height, width = img.shape[:2]

    # Если изображение уже меньше или равно max_dimension по обеим сторонам
    if max(width, height) <= max_dimension:
        # Конвертируем BGR -> RGB перед возвратом
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Вычисляем новые размеры с сохранением пропорций
    if width > height:
        new_width = max_dimension
        new_height = int(height * max_dimension / width)
    else:
        new_height = max_dimension
        new_width = int(width * max_dimension / height)

    # Ресайз с интерполяцией LANCZOS (в OpenCV - INTER_LANCZOS4)
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    logger.info(
        f"Изображение {image_path.name} уменьшено с {width}x{height} до {new_width}x{new_height}"
    )

    # Конвертируем BGR -> RGB перед возвратом
    return cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
