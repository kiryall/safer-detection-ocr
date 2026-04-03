import random
from pathlib import Path
from typing import List, Dict, Any

from configs.config import SUPPORTED_IMAGE_FORMATS
from src.detection.yolo_detector import YOLODetector
from src.preprocessing.image_utils import ImageCropper
from src.utils.logger import setup_logging

# Заглушки для будущих модулей
from src.ocr.ocr_engine import OCREngine  # будет реализовано позже
from src.postprocessing.text_cleaner import clean_text
from src.renaming.renamer import FileRenamer

logger = setup_logging(log_file="batch_processor.log")


class BatchProcessor:
    """
    Основной класс обработки пачки фотографий.
    Сейчас: Детекция + Кроп
    Позже:   + OCR + Очистка + Переименование + Группировка
    """

    def __init__(self):
        self.detector = YOLODetector()
        self.cropper = ImageCropper(save_crops=True)

        # Заглушки для будущих компонентов
        self.ocr_engine = OCREngine()  # пока заглушка
        self.renamer = FileRenamer()  # пока заглушка

    def process_single_image(self, image_path: Path) -> Dict[str, Any]:
        """Полная обработка одного изображения (с заглушками)."""
        logger.info(f"Обработка: {image_path.name}")

        # 1. Детекция
        det_result = self.detector.detect_single(image_path)

        record = {
            "filename": image_path.name,
            "original_path": str(image_path),
            "detection_success": det_result["success"],
            "detection_confidence": 0.0,
            "bbox": None,
            "crop": None,
            "crop_path": None,
            "recognized_text": None,
            "text_confidence": 0.0,
            "clean_text": None,
            "final_name": None,
            "status": "failed",
            "message": det_result["message"],
        }

        if not det_result["success"] or not det_result.get("best_detection"):
            return record

        # 2. Кроп
        best_det = det_result["best_detection"]
        bbox = best_det["bbox"]
        conf = best_det["confidence"]

        crop = self.cropper.crop_and_save(image_path, bbox, conf)

        if crop is None:
            record["message"] = "Не удалось обрезать табличку"
            return record

        record.update(
            {
                "detection_confidence": conf,
                "bbox": bbox,
                "crop": crop,  # PIL Image
                "crop_path": crop,
            }
        )

        # Set status to success after crop succeeds
        record["status"] = "success"

        # === ЗАГЛУШКИ ДЛЯ БУДУЩИХ ФИЧ ===

        # 3. OCR (заглушка)
        try:
            text, text_conf = self.ocr_engine.recognize(crop)
            record["recognized_text"] = text
            record["text_confidence"] = text_conf
        except Exception as e:
            record["message"] += f" | OCR error: {e}"

        # 4. Очистка текста
        if record["recognized_text"]:
            record["clean_text"] = clean_text(record["recognized_text"])

        # 5. Переименование
        if record["clean_text"]:
            try:
                record["final_name"] = self.renamer.generate_name(
                    original_path=image_path, number=record["clean_text"]
                )
            except Exception as e:
                record["message"] += f" | Renaming error: {e}"

        return record

    def process_folder(
        self, input_dir: Path, limit: int = None
    ) -> List[Dict[str, Any]]:
        """Обработка всей папки."""
        # Поддерживаемые форматы изображений
        supported = SUPPORTED_IMAGE_FORMATS

        input_dir = Path(input_dir)
        # Получаем список всех изображений в папке, поддерживаемых форматов
        image_paths = [
            p
            for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in supported
        ]

        if limit:
            image_paths = random.sample(image_paths, min(limit, len(image_paths)))

        logger.info(f"Найдено {len(image_paths)} изображений для обработки.")

        results = []
        for i, path in enumerate(image_paths, 1):
            result = self.process_single_image(path)
            results.append(result)

            if i % 50 == 0 or i == len(image_paths):
                success = sum(1 for r in results if r["status"] == "success")
                logger.info(f"Обработано {i}/{len(image_paths)} | Успешно: {success}")

        return results
