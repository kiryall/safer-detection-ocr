# src\detection\yolo_detector.py

from pathlib import Path
from typing import Tuple, Dict, Optional
import cv2
import numpy as np
import csv
import os
from ultralytics import YOLO
from PIL import Image
from src.utils.logger import setup_logging

from configs.config import OUTPUT_CROPS_DIR, YOLO_BEST_MODEL, OUTPUT_DIR


logger = setup_logging(log_file="yolo_detector.log", console=True, remove_file=True, logger_name="yolo_detector")


class YOLODetector:
    """
    Класс для детекции таблички.
    Сохроаняет кропы на диск.
    """

    def __init__(
            self,
            model_path: str = YOLO_BEST_MODEL,
            conf_threshold: float = 0.5,
            iou_threshold: float = 0.5,
            min_confidence: float = 0.4,
            save_crops: bool = True,
            crops_dir: Optional[Path] = OUTPUT_CROPS_DIR
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_confidence = min_confidence
        self.save_crops = save_crops
        self.crops_dir = crops_dir
        self.metadata_dir = OUTPUT_DIR / "metadata"
        if self.save_crops:
            self.crops_dir.mkdir(parents=True, exist_ok=True)
            self.success_dir = self.crops_dir / "success"
            self.low_conf_dir = self.crops_dir / "low_confidence"
            self.no_plate_dir = self.crops_dir / "no_plate"
            for d in [self.success_dir, self.low_conf_dir, self.no_plate_dir]:
                d.mkdir(parents=True, exist_ok=True)
        
        # Создаем директорию для метаданных
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _save_crop(self,
                   cropped_pil: Image.Image,
                   original_stem: str,
                   conf: float,
                   success: bool,
                   original_filename: str) -> Path:
        
        """
        Сохраняет кроп на диск в зависимости от результата детекции.
        Args:
            cropped_pil: PIL Image кропа таблички.
            original_stem: stem оригинального изображения для формирования имени файла.
            confidence: confidence детекции для формирования имени файла.
            success: флаг успешной детекции для выбора директории сохранения.
            original_filename: оригинальное имя файла для сохранения без изменений.
        Returns:
            Путь к сохраненному кропу.
        """

        if not self.save_crops:
            logger.info("Сохранение кропов отключено. Кроп не будет сохранен.")
            return None
        
        # Используем оригинальное имя файла для сохранения кропа
        if success:
            save_dir = self.success_dir
            logger.info(f"Успешная детекция: {original_stem}, confidence: {conf:.3f}. Сохранение кропа в {save_dir}.")
        else:
            save_dir = self.low_conf_dir if conf > 0 else self.no_plate_dir
            logger.info(f"Низкая уверенность или отсутствие таблички: {original_stem}, confidence: {conf:.3f}. Сохранение кропа в {save_dir}.")

        # Сохраняем кроп с оригинальным именем файла
        save_path = save_dir / original_filename
        cropped_pil.save(save_path, quality=95)
        logger.info(f"Кроп сохранен: {save_path}")
        
        return save_path

    def _save_metadata(self, image_path: str, confidence: float, has_tablet: bool, crop_path: Optional[str] = None):
        """
        Сохраняет метаданные в CSV файл.
        Args:
            image_path: путь к исходному изображению
            confidence: уверенность модели
            has_tablet: наличие таблички (True/False)
            crop_path: путь к сохраненному кропу (если был создан)
        """
        image_path_obj = Path(image_path)
        csv_path = self.metadata_dir / "crop_output.csv"
        
        # Подготовим данные для записи
        row_data = {
            "filename": image_path_obj.name,
            "confidence": confidence,
            "has_tablet": has_tablet,
            "crop_path": str(crop_path) if crop_path is not None else None
        }
        
        # Проверим, существует ли файл и нужно ли писать заголовки
        write_header = not csv_path.exists()
        
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 'confidence', 'has_tablet', 'crop_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if write_header:
                writer.writeheader()  # Записываем заголовки только в первый раз
            
            writer.writerow(row_data)
        
        logger.info(f"Метаданные сохранены в CSV: {csv_path}")

    def detect(
            self, image_path: str) -> Dict:
        """
        Детектирует табличку на изображении, сохраняет кропы и возвращает результаты.
        Args:
            image_path: Путь к изображению для детекции.
        Returns:
            Словарь с результатами детекции, включая координаты, confidence и путь к сохраненному кропу.
        """
        image_path = Path(image_path)
        stem = image_path.stem

        try:
            results = self.model.predict(
                source=str(image_path),
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                max_det=5       
            )
            
            detections = []
            img_pil = Image.open(image_path).convert("RGB")
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    cropped_cv = img_cv[y1:y2, x1:x2]
                    if cropped_cv.size == 0:
                        continue

                    cropped_pil = Image.fromarray(cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2RGB))

                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "cropped_image": cropped_pil,
                        "crop_path": None,          # заполним ниже, если сохраняем
                        "image_path": str(image_path)
                    })

            detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)

            if not detections:
                message = "Табличка не обнаружена"
                logger.info(f"{message} на изображении: {image_path}")
                
                # Сохраняем метаданные даже если табличка не найдена
                self._save_metadata(
                    image_path=image_path,
                    confidence=0.0,
                    has_tablet=False
                )
                
                return {
                    "success": False,
                    "detections": [],
                    "best_detection": None,
                    "message": message
                }

            best = detections[0]
            # Передаем оригинальное имя файла для сохранения кропа
            crop_path = self._save_crop(best["cropped_image"], stem, best["confidence"],
                                       best["confidence"] >= self.min_confidence,
                                       Path(image_path).name)

            if crop_path:
                best["crop_path"] = str(crop_path)

            success = best["confidence"] >= self.min_confidence
            
            # Сохраняем метаданные
            self._save_metadata(
                image_path=image_path,
                confidence=best["confidence"],
                has_tablet=success,
                crop_path=crop_path
            )

            return {
                "success": success,
                "detections": detections,
                "best_detection": best,
                "message": f"Табличка {'успешно' if success else 'сомнительно'} обнаружена (conf: {best['confidence']:.3f})"
            }

        except Exception as e:
            logger.error(f"Ошибка при детекции на изображении {image_path}: {str(e)}", exc_info=True)
            
            # Сохраняем метаданные даже в случае ошибки
            self._save_metadata(
                image_path=image_path,
                confidence=0.0,
                has_tablet=False
            )
            
            return {
                "success": False,
                "detections": [],
                "best_detection": None,
                "message": f"Ошибка детекции: {str(e)}"
            }
        

    def get_best_crop(self, image_path: str) -> Tuple[Optional[Image.Image], float, str, Optional[str]]:
        """Возвращает: cropped_image, confidence, message, crop_path"""
        result = self.detect(image_path)
        if result["best_detection"]:
            best = result["best_detection"]
            return (
                best["cropped_image"],
                best["confidence"],
                result["message"],
                best.get("crop_path")
            )
        return None, 0.0, result["message"], None