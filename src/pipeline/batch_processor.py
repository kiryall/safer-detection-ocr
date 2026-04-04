# src/pipeline/batch_processor.py

import random
from pathlib import Path
from typing import List, Dict, Any

from configs.config import SUPPORTED_IMAGE_FORMATS
from src.detection.yolo_detector import YOLODetector
from src.preprocessing.image_utils import ImageCropper
from src.ocr.ocr_engine import OCREngine
from src.postprocessing.text_cleaner import clean_text
from src.renaming.renamer import FileRenamer
from src.utils.logger import setup_logging

logger = setup_logging(log_file="batch_processor.log")


class BatchProcessor:
    """
    Основной класс обработки фотографий.
    Сейчас: Детекция → Кроп → OCR → Очистка → Переименование
    """

    def __init__(self, prefix: str = "SEFER"):
        self.detector = YOLODetector()
        self.cropper = ImageCropper(save_crops=True)
        self.ocr_engine = OCREngine(gpu=False)
        self.renamer = FileRenamer(default_prefix=prefix)

    def process_single_image(self, image_path: Path) -> Dict[str, Any]:
        """Полная обработка одного изображения."""
        logger.info(f"Обработка: {image_path.name}")

        record = {
            "filename": image_path.name,
            "original_path": str(image_path),
            "detection_success": False,
            "detection_confidence": 0.0,
            "bbox": None,
            "crop_path": None,
            "recognized_text": None,
            "text_confidence": 0.0,
            "clean_text": None,
            "final_name": None,
            "status": "failed",
            "message": "",
        }

        # 1. Детекция
        det_result = self.detector.detect_single(image_path)

        record["detection_success"] = det_result["success"]
        record["message"] = det_result["message"]

        if not det_result["success"] or not det_result.get("best_detection"):
            record["status"] = "failed_detection"
            return record

        best_det = det_result["best_detection"]
        bbox = best_det["bbox"]
        det_conf = best_det["confidence"]

        record["detection_confidence"] = det_conf
        record["bbox"] = bbox

        # 2. Кроп
        crop, cropped_path = self.cropper.crop_and_save(image_path, bbox, det_conf)
        if crop is None:
            record["message"] += " | Не удалось создать кроп"
            record["status"] = "failed_crop"
            return record

        record["crop_path"] = str(cropped_path)

        # 3. OCR
        text, text_conf = self.ocr_engine.recognize(crop, visualize=True, image_name=image_path.stem)

        record["recognized_text"] = text
        record["text_confidence"] = text_conf
        record["clean_text"] = clean_text(text) if text else None
        
        # 4. Очистка текста
        if text:
            record["clean_text"] = text # временно сохраняем неочищенный текст для отладки

        # 5. Генерация нового имени
        if record["clean_text"]:
            record["final_name"] = self.renamer.generate_name_with_counter(
                image_path, record["clean_text"]
            )
            record["status"] = "success"
            record["message"] = f"Успешно | Детекция: {det_conf:.3f} | OCR: {text_conf:.3f}"
        else:
            if det_conf >= 0.65:
                record["status"] = "partial_ocr_failed"
            elif det_conf >= 0.40:
                record["status"] = "low_confidence"
            else:
                record["status"] = "very_low_confidence"
            
            record["message"] = f"Детекция: {det_conf:.3f} | OCR не распознал номер"

        return record

    def process_folder(
        self, input_dir: Path, limit: int = None, shuffle: bool = True
    ) -> List[Dict[str, Any]]:
        """Обработка всей папки (с опциональным перемешиванием)."""
        input_dir = Path(input_dir)

        image_paths = [
            p for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_FORMATS
        ]

        if not image_paths:
            logger.warning(f"В папке {input_dir} не найдено поддерживаемых изображений.")
            return []

        if limit:
            if shuffle:
                image_paths = random.sample(image_paths, min(limit, len(image_paths)))
            else:
                image_paths = image_paths[:limit]

        logger.info(f"Найдено {len(image_paths)} изображений для обработки.")

        results = []
        for i, path in enumerate(image_paths, 1):
            result = self.process_single_image(path)
            results.append(result)

            if i % 50 == 0 or i == len(image_paths):
                success = sum(1 for r in results if r["status"] == "success")
                low = sum(1 for r in results if "low" in r["status"])
                logger.info(
                    f"Обработано {i}/{len(image_paths)} | "
                    f"Успешно: {success} | Низкая уверенность: {low}"
                )

        return results