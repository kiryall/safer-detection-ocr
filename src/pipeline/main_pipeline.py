# src/pipeline/main_pipeline.py

from pathlib import Path
from typing import Dict

from configs.config import OUTPUT_CROPS_DIR, OUTPUT_DIR, RAW_DATA_DIR, YOLO_BEST_MODEL
from src.detection.yolo_detector import YOLODetector
from src.preprocessing.image_loader import load_image
from src.reporting.report_generator import ReportGenerator
from src.utils.logger import setup_logging

logger = setup_logging(log_file="main_pipeline.log", remove_file=True, console=True, logger_name="main_pipeline")


class SaferPipeline:
    """
    Главный класс для запуска всего пайплайна.
    """

    def __init__(
            self,
            model_path: str = YOLO_BEST_MODEL,
            conf_threshold: float = 0.5,
            min_confidence: float = 0.4,
            save_crops: bool = True,       
    ):
        self.detector = YOLODetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            min_confidence=min_confidence,
            save_crops=save_crops,
            crops_dir=OUTPUT_CROPS_DIR
        )

        self.report_generator = ReportGenerator(output_dir=OUTPUT_DIR)
        self.processed_count = 0
        self.success_count = 0

    def process_single_image(self, image_path: Path) -> Dict:
        """
        Обрабатывает одно изображение: загружает, детектирует, сохраняет кропы и генерирует отчет.
        Args:
            image_path: путь к изображению для обработки.
        Returns:
            Результат обработки в виде словаря.
        """

        logger.info(f"Начало обработки изображения: {image_path}")

        try:
            result = self.detector.detect(str(image_path))

            record = {
                "filename": image_path.name,
                "original_path": str(image_path),
                "success": result["success"],
                "confidence": result["best_detection"]["confidence"] if result["best_detection"] else 0.0,
                "message": result["message"],
                "crop_path": result["best_detection"].get("crop_path") if result["best_detection"] else None,
                "bbox": result["best_detection"]["bbox"] if result["best_detection"] else None,
            }

            if result["success"]:
                self.success_count += 1

            self.processed_count += 1

            return record

        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {e}")
            return {
                "filename": image_path.name,
                "original_path": str(image_path),
                "success": False,
                "confidence": 0.0,
                "message": f"Ошибка обработки: {str(e)}",
                "crop_path": None,
                "bbox": None,
            }