# src/pipeline/batch_processor.py

import logging
import random
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from PIL import Image

from configs.config import (
    CONF_THRESHOLD,
    CONFIDANCE_LEVEL_THRESHOLD,
    DEBUG_SAVE_CROPS,
    LOGGING_CONSOLE,
    LOW_CONFIDENCE_DIR,
    OCR_ACCEPT_THRESHOLD,
    OCR_THRESHOLD,
    OUTPUT_DIR,
    SEED,
    SUCCESS_DIR,
    SUPPORTED_IMAGE_FORMATS,
    VERY_LOW_DIR,
    YOLO_CONF_THRESHOLD,
)

from src.detection.yolo_detector import YOLODetector
from src.ocr.ocr_engine import OCREngine
from src.postprocessing.text_cleaner import clean_validate_text
from src.preprocessing.image_utils import ImageCropper
from src.renaming.renamer import FileRenamer
from src.utils.logger import setup_logging

logger = setup_logging(log_file="batch_processor.log",
                        console=LOGGING_CONSOLE,
                        remove_file=True,
                        logger_name="batch_processor",
                        level=logging.DEBUG,)


class BatchProcessor:
    """
    Основной класс обработки фотографий.
    Детекция → Кроп → OCR → Очистка → Переименование
    """

    def __init__(
        self,
        prefix: str = "SEFER",
        visualize: bool = False,
        base_output_dir: Optional[Path] = None,
        conf_threshold: float = None,
        yolo_conf_threshold: float = None,
        ocr_conf_threshold: float = None,          
        ocr_accept_threshold: float = None,
        confidence_level_threshold: float = None,
        debug_save_crops: bool = False,
        ):
        
        self.prefix = prefix
        self.visualize = visualize
        self.conf_threshold = conf_threshold or CONF_THRESHOLD
        self.ocr_conf_threshold = ocr_conf_threshold or OCR_THRESHOLD
        self.ocr_accept_threshold = ocr_accept_threshold or OCR_ACCEPT_THRESHOLD
        self.confidence_level_threshold = confidence_level_threshold or CONFIDANCE_LEVEL_THRESHOLD
        self.debug_save_crops = debug_save_crops or DEBUG_SAVE_CROPS

        # === Динамические папки ===
        self.base_output_dir = Path(base_output_dir) if base_output_dir else OUTPUT_DIR
        self.visualizations_dir = self.base_output_dir / "visualizations"
        self.success_dir = self.base_output_dir / "success"
        self.low_conf_dir = self.base_output_dir / "low_confidence"
        self.very_low_dir = self.base_output_dir / "very_low"
        self.crops_dir = self.base_output_dir / "crops"

        for d in (
            self.success_dir,
            self.low_conf_dir,
            self.very_low_dir,
            self.crops_dir,
            self.visualizations_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

        # Детектор с кастомным порогом
        self.detector = YOLODetector(
                                    conf_threshold=yolo_conf_threshold or YOLO_CONF_THRESHOLD
                                )

        self.cropper = ImageCropper(save_crops=self.debug_save_crops,
                                    crops_base_dir=self.crops_dir,
                                    conf_threshold=self.conf_threshold
                                    )
        self.ocr_engine = OCREngine(gpu=False,
                                    conf_threshold=self.ocr_conf_threshold,           # технический
                                    ocr_accept_threshold=self.ocr_accept_threshold,     # бизнес-порог,
                                    output_vis_dir=self.visualizations_dir,
                                    )
        self.renamer = FileRenamer(default_prefix=prefix)

    def process_single_image(self, image_path: Path) -> Dict[str, Any]:
        """Полная обработка одного изображения."""
        logger.info(f"Обработка: {image_path.name}")

        record = {
            "filename": image_path.name,
            "original_path": str(image_path),
            "detection_confidence": 0.0,
            "crop_path": None,           # debug only
            "recognized_text": None,
            "text_confidence": 0.0,
            "clean_text": None,
            "final_name": None,
            "final_path": None,
            "status": "very_low_confidence",
            "message": "",
            'confidence_level': 0.0
        }

        # 1. Детекция
        det_result = self.detector.detect_single(image_path)

        best_det = det_result.get("best_detection")
        det_conf = best_det["confidence"] if best_det else 0.0
        record["detection_confidence"] = det_conf

        # === КЛЮЧЕВАЯ ЛОГИКА ===
        if (not det_result.get("success") or 
            not best_det or 
            det_conf < self.conf_threshold):

            record["message"] = f"Детекция ниже порога ({det_conf:.3f} < {self.conf_threshold:.3f})"
            record["final_path"] = self._save_final_file(
                image_path, image_path, "very_low", keep_name=True
            )
            return record

        # Если дошли сюда — детекция считается успешной
        logger.info(f"Детекция прошла порог: {det_conf:.3f} >= {self.conf_threshold:.3f}")

        # 2. Кроп
        crop, cropped_path = self.cropper.crop_obb(
            image_path=image_path,
            obb=best_det["obb"],           # формат [cx, cy, w, h, angle_rad]
            confidence=det_conf,
            
        )

        if crop is None:
            record["message"] = "Не распознано. Не удалось создать кроп"
            record["final_path"] = self._save_final_file(image_path, image_path, "very_low", keep_name=True)
            return record

        record["crop_path"] = str(cropped_path) if cropped_path else None

        # 3. OCR
        text, text_conf = self.ocr_engine.recognize(
            crop, visualize=self.visualize, image_name=image_path.stem
        )

        record["recognized_text"] = text
        record["text_confidence"] = text_conf
        record["clean_text"] = clean_validate_text(text) if text else None
        clean_text = record["clean_text"]

        # 4. Очистка текста и проверка на валидность для моделей OCR
        # Сейчас не используем

        # 5. Определяем группу и сохраняем финальный файл
        # Считаем общую уверенность
        confidence_level = det_conf * text_conf
        record['confidence_level'] = round(confidence_level, 2)
        
        if clean_text:
            # сравниваем уверенность с порогом
            if confidence_level >= self.confidence_level_threshold:
                record["status"] = "success"
                record["final_name"] = self.renamer.generate_name_with_counter(image_path, clean_text)
                record["final_path"] = self._save_final_file(
                    image_path, image_path, "success", final_name=record["final_name"]
                )
            else:
                record["status"] = "low_confidence"
                record["final_name"] = self.renamer.generate_name_with_counter(
                    image_path, clean_text, low_confidence=True
                )
                record["final_path"] = self._save_final_file(
                    image_path, image_path, "low_confidence", final_name=record["final_name"]
                )
        else:
            record["status"] = "very_low_confidence"
            record["message"] = f"Детекция OK ({det_conf:.3f}), но OCR не прочитал номер"
            record["final_path"] = self._save_final_file(
                image_path, image_path, "very_low", keep_name=True
            )

        record["message"] = record.get("message", "") + f" | Детекция: {det_conf:.3f} | OCR: {text_conf:.3f}"
        return record

    def process_folder(
        self,
        input_dir: Path,
        limit: int = None,
        shuffle: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Обработка всей папки (с опциональным перемешиванием)."""
        
        input_dir = Path(input_dir)
        image_paths = [
            p
            for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_FORMATS
        ]

        if not image_paths:
            logger.warning(
                f"В папке {input_dir} не найдено поддерживаемых изображений."
            )
            return []

        # Опционально ограничиваем количество и перемешиваем
        if limit:
            if shuffle:
                rnd = random.Random(SEED)
                image_paths = rnd.sample(image_paths, min(limit, len(image_paths)))
            else:
                image_paths = image_paths[:limit]

        logger.info(f"Найдено {len(image_paths)} изображений для обработки.")

        results = []
        for i, path in enumerate(image_paths, 1):
            result = self.process_single_image(path)
            results.append(result)

            if progress_callback:
                progress_callback(i, len(image_paths))

            if i % 50 == 0 or i == len(image_paths):
                success = sum(1 for r in results if r["status"] == "success")
                low = sum(1 for r in results if "low" in r["status"])
                logger.info(
                    f"Обработано {i}/{len(image_paths)} | "
                    f"Успешно: {success} | Низкая уверенность: {low}"
                )

        return results
    
    def _save_final_file(
        self,
        original_path: Path,
        source: Union[Path, Image.Image],   # Path = оригинал, Image = кроп
        target_group: str,                  # "success" / "low_confidence" / "very_low"
        final_name: str = None,
        keep_name: bool = False
    ) -> Path:
        """Единственная запись на диск в конце пайплайна"""
        if target_group == "success":
            target_dir = SUCCESS_DIR
        elif target_group == "low_confidence":
            target_dir = LOW_CONFIDENCE_DIR
        else:
            target_dir = VERY_LOW_DIR

        if keep_name:
            target_path = target_dir / original_path.name
        else:
            target_path = target_dir / final_name

        if isinstance(source, Image.Image):
            source.save(target_path, quality=95)
        else:
            # копируем оригинал
            shutil.copy2(source, target_path)

        logger.info(f"Финальный файл сохранён → {target_path}")
        return target_path

    def _detection():
        pass

    def _crop():
        pass

    def _ocr():
        pass
