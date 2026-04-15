# src\detection\yolo_detector.py

from pathlib import Path
from typing import Any, Dict, List, Union

from ultralytics import YOLO

from configs.config import (
    LOGGING_CONSOLE,
    YOLO_BEST_MODEL,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
)
from src.preprocessing.image_utils import resize_image_if_large
from src.utils.logger import setup_logging

logger = setup_logging(
    log_file="yolo_detector.log",
    console=LOGGING_CONSOLE,
    remove_file=True,
    logger_name="yolo_detector",
)


class YOLODetector:
    """
    Детекции таблички.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = YOLO_BEST_MODEL,
        conf_threshold: float = YOLO_CONF_THRESHOLD,  # Порог confidence для детекции
        iou_threshold: float = YOLO_IOU_THRESHOLD,  # Порог IoU для NMS
        max_det: int = 5,  # Максимальное количество детекций на изображение
    ):
        self.model = YOLO(str(model_path))
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det

    def detect_single(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Детектирует табличку на изображении.
        Args:
            image_path: Путь к изображению для детекции.
        Returns:
            Словарь с результатами детекции, включая координаты, confidence.
        """
        image_path = Path(image_path)
        logger.info(f"Детекция изображения: {image_path}")

        try:
            # Получаем размеры оригинального изображения
            from PIL import Image

            original_img = Image.open(image_path)
            original_width, original_height = original_img.size

            # Загружаем и при необходимости уменьшаем изображение
            image_array = resize_image_if_large(image_path)
            resized_height, resized_width = image_array.shape[
                :2
            ]  # shape[0]=h, shape[1]=w

            # Вычисляем коэффициенты масштабирования
            scale_x = original_width / resized_width
            scale_y = original_height / resized_height

            # Выполняем детекцию с помощью модели YOLO
            results = self.model.predict(
                source=image_array,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_det,
                verbose=False,
                task="obb",
            )

            # Собираем все детекции в список
            detections = []

            # Проходим по всем результатам (может быть несколько, если модель возвращает несколько слоев)
            for result in results:
                if not hasattr(result, "obb") or result.obb is None:
                    continue

                for obb in result.obb:
                    data = obb.data.tolist()[0]
                    cx, cy, w, h, angle_rad, conf, cls = data

                    # Масштабируем координаты обратно к оригинальному размеру
                    cx *= scale_x
                    cy *= scale_y
                    w *= scale_x
                    h *= scale_y

                    detections.append(
                        {
                            "type": "obb",
                            "obb": [
                                cx,
                                cy,
                                w,
                                h,
                                angle_rad,
                            ],  # центр, размеры, угол в радианах
                            "confidence": float(conf),
                        }
                    )

            if not detections:
                return {
                    "success": False,
                    "detections": [],
                    "best_detection": None,
                    "message": "Табличка не обнаружена",
                }

            detections.sort(key=lambda x: x["confidence"], reverse=True)
            best = detections[0]

            return {
                "success": True,
                "detections": detections,
                "best_detection": best,
                "message": f"Найдено {len(detections)} объектов, conf: {best['confidence']:.3f}",
            }

        except Exception as e:
            logger.error(f"Ошибка детекции YOLO OBB: {e}")
            return {
                "success": False,
                "detections": [],
                "best_detection": None,
                "message": f"Ошибка при детекции: {e}",
            }

    def detect_batch(self, image_paths: List[str | Path]) -> List[Dict[str, Any]]:
        """Детекция списка изображений.
        Args:
            image_paths: Список путей к изображениям для детекции.
        Returns:
            Список словарей с результатами детекции для каждого изображения."""
        return [self.detect_single(path) for path in image_paths]
