# src/ocr/ocr_engine.py

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import logging

from configs.config import (ALLOWLIST,
                            DIGIT_DETECTOR_MODEL,
                            LOGGING_CONSOLE,
                            OCR_ACCEPT_THRESHOLD,
                            OCR_IOU_THRESHOLD,
                            OCR_MAX_DET,
                            OCR_THRESHOLD,
                            )
from src.utils.logger import setup_logging
from src.utils.ocr_utils import save_ocr_visualization
from ultralytics import YOLO

logger = setup_logging(
    log_file="ocr_engine.log",
    console=LOGGING_CONSOLE,
    remove_file=True,
    logger_name="ocr_engine",
    level=logging.DEBUG,
)


class OCREngine:
    """OCR движок для распознавания текста на табличках."""

    def __init__(
        self,
        gpu: bool = False,
        model_path: Path = Path(DIGIT_DETECTOR_MODEL),
        conf_threshold: float = OCR_THRESHOLD,                    # технический порог для модели
        ocr_accept_threshold: float = OCR_ACCEPT_THRESHOLD,       # бизнес-порог принятия результата
        allowlist: str = ALLOWLIST,
        iou_threshold: float = OCR_IOU_THRESHOLD,
        max_det: int = OCR_MAX_DET,
        output_vis_dir: Optional[Path] = None,
    ):
        self.model = YOLO(str(model_path))
        self.conf_threshold = conf_threshold             # используется только в predict
        self.ocr_accept_threshold = ocr_accept_threshold # порог, ниже которого считаем OCR неудачным
        self.allowlist = allowlist
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.output_vis_dir = Path(output_vis_dir) if output_vis_dir else None

    def recognize(
        self,
        cropped_image: Image.Image,
        visualize: bool = False,
        image_name: Optional[str] = None
    ) -> Tuple:
        """Распознаёт текст
        Args:
            cropped_image: Обрезанное изображение в формате PIL или None.
            image_name: Имя файла для сохранения визуализации.
            visualize: Флаг для сохранения визуализации.
        Returns:
            Кортеж (распознанный текст, confidence) или (None, 0.0) если распознавание не удалось.
        """
        if cropped_image is None:
            return None, 0.0
        logger.info(f"Детекция номера: {cropped_image}")

        try:
            if cropped_image.mode != "RGB":
                cropped_image = cropped_image.convert("RGB")

            image_array = np.asarray(cropped_image, dtype=np.uint8)

            results = self.model.predict(
                source=image_array,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_det,
                verbose=False,
            )

            if not results or not hasattr(results[0], "boxes") or results[0].boxes is None:
                logger.warning("Объекты не обнаружены")
                return None, 0.0

            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
            confidences = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
            classes = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else []

            if len(boxes) == 0:
                return None, 0.0

            detections = []
            allowlist = self.allowlist

            for box, conf, detector_class in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2
                cls_idx = int(detector_class)
                char = allowlist[cls_idx] if cls_idx < len(allowlist) else "?"
                detections.append({
                    "box": box,
                    "center_x": center_x,
                    "char": char,
                    "conf": conf,
                    "cls": cls_idx,
                })

            # Сортируем символы слева направо
            detections.sort(key=lambda x: x["center_x"])

            recognized_text = "".join(d["char"] for d in detections)
            avg_conf = sum(d["conf"] for d in detections) / len(detections) if detections else 0.0

            # === КЛЮЧЕВАЯ ЛОГИКА ===
            if avg_conf < self.ocr_accept_threshold:
                logger.info(
                    f"[{image_name}] OCR confidence слишком низкий: {avg_conf:.3f} "
                    f"< {self.ocr_accept_threshold:.3f} → считаем нераспознанным"
                )
                if visualize:
                    save_ocr_visualization(
                        cropped_image,
                        detections,
                        image_name,
                        avg_conf,
                        logger,
                        output_dir=self.output_vis_dir,
                    )
                return None, avg_conf   # Возвращаем None, даже если текст распознан

            # Успешное распознавание
            logger.info(f"[{image_name}] OCR успешно: '{recognized_text}' (conf={avg_conf:.3f})")

            if visualize:
                save_ocr_visualization(
                    cropped_image,
                    detections,
                    image_name,
                    avg_conf,
                    logger,
                    output_dir=self.output_vis_dir,
                )

            return recognized_text, avg_conf

        except Exception as e:
            logger.error(f"Ошибка OCR: {e}")
            return None, 0.0

    def _clean_ocr_text(self, text: str) -> Optional[str]:
        """Очистка и нормализация распознанного текста."""
        ...
