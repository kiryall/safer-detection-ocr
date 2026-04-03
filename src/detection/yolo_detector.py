# src\detection\yolo_detector.py

from pathlib import Path
from typing import Dict, Any, List
from ultralytics import YOLO
from src.utils.logger import setup_logging

from configs.config import YOLO_BEST_MODEL, YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD


logger = setup_logging(log_file="yolo_detector.log", console=True, remove_file=True, logger_name="yolo_detector")


class YOLODetector:
    """
    Детекции таблички.
    """

    def __init__(
            self,
            model_path: str = YOLO_BEST_MODEL,
            conf_threshold: float = YOLO_CONF_THRESHOLD, # Порог confidence для детекции
            iou_threshold: float = YOLO_IOU_THRESHOLD, # Порог IoU для NMS
            max_det: int = 5 # Максимальное количество детекций на изображение
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold        
        self.max_det = max_det


    def detect_single(self, image_path: Path) -> Dict[str, Any]:
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
            # Выполняем детекцию с помощью модели YOLO
            results = self.model.predict(
                source=str(image_path),
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_det,
                verbose=False,
            )

            # Собираем все детекции в список
            detections = []

            # Проходим по всем результатам (может быть несколько, если модель возвращает несколько слоев)
            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])

                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                    })

            # Сортируем детекции по confidence в порядке убывания
            detections.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Если нет детекций, возвращаем результат с сообщением об отсутствии таблички
            if not detections:
                return {
                    "success": False,
                    "detections": [],
                    "best_detection": None,
                    "message": "Табличка не обнаружена",
                }
            
            best = detections[0]

            # Определяем успешность детекции на основе порога confidence
            success = best["confidence"] >= self.conf_threshold
            
            return {
                "success": success,
                "detections": detections,
                "best_detection": best,
                "message": f"Найдено {len(detections)} объектов, лучшая conf: {best['confidence']:.3f}",
            }

        except Exception as e:
            return {
                "success": False,
                "detections": [],
                "best_detection": None,
                "message": f"Ошибка при детекции YOLO: {str(e)}",
            }
        
    def detect_batch(self, image_paths: List[str | Path]) -> List[Dict[str, Any]]:
        """Детекция списка изображений.
        Args:
            image_paths: Список путей к изображениям для детекции.
        Returns:
            Список словарей с результатами детекции для каждого изображения."""
        return [self.detect_single(path) for path in image_paths]