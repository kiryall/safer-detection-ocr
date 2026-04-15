# src/ocr/paddleocr_engine.py
# Движок OCR на основе PaddleOCR для распознавания текста на табличках.
# ТРЕБУЕТСЯ ДОРАБОТКА: сейчас он не используется в основном пайплайне, но может быть полезен для сравнения с YOLO OCR.

import io
import logging
import os
import sys
import warnings

from configs.config import OCR_MODEL_PATH, LOGGING_CONSOLE

# Отключаем предупреждения Paddle до импорта
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "False"
os.environ["PADDLE_DISABLE_PRINT_SIGNAL"] = (
    "1"  # Отключает некоторые сигналы печати Paddle
)

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

from src.utils.logger import setup_logging
from src.utils.ocr_utils import save_paddleocr_visualization
from src.utils.yaml_loader import load_config

# Подавляем логи paddle и paddlex ДО создания PaddleOCR
logging.getLogger("paddle").setLevel(logging.ERROR)
logging.getLogger("paddlex").setLevel(logging.ERROR)
logging.getLogger("ppocr").setLevel(logging.ERROR)

logger = setup_logging(
    log_file="paddleocr_engine.log",
    console=LOGGING_CONSOLE,
    logger_name="paddleocr_engine",
    level=logging.DEBUG,
)


class PaddleOCREngine:
    """PaddleOCR движок для распознавания текста на табличках."""

    def __init__(
        self,
        gpu: bool = False,
        config_path: Path = Path("configs\paddleocr_config.yaml"),
        models_dir: Optional[Path] = OCR_MODEL_PATH,
        output_vis_dir: Optional[Path] = None,
    ):
        self.models_dir = Path(models_dir)
        self.config = load_config(config_path)
        self.allowlist = self.config["allowlist"]
        # PaddleOCR
        self.ocr = PaddleOCR(
            lang="en",
            device="gpu" if gpu else "cpu",
            use_angle_cls=True,
            det_db_thresh=0.3,
            rec_batch_num=6,
        )

        logger.info(f"PaddleOCR инициализирован (GPU: {gpu})")
        self.output_vis_dir = Path(output_vis_dir) if output_vis_dir else None

    def recognize(
        self, cropped_image, image_name: str = "crop", visualize: bool = False
    ) -> Tuple[Optional[str], float]:
        """Распознаёт текст на кропе и возвращает текст и среднюю confidence."""

        if cropped_image is None:
            return None, 0.0

        if isinstance(cropped_image, tuple):
            cropped_image = cropped_image[0] if cropped_image else None

        if not isinstance(cropped_image, Image.Image):
            logger.error(f"Неверный тип: {type(cropped_image)}")
            return None, 0.0

        try:
            img = np.asarray(cropped_image.convert("RGB"), dtype=np.uint8)

            result = self.ocr.predict(
                img,
                return_word_box=True,
            )

            if not result:
                return None, 0.0

            all_texts = []
            total_conf = 0.0
            count = 0

            for res in result:
                texts = res.get("rec_texts", [])
                scores = res.get("rec_scores", [])
                boxes = res.get("dt_polys", [])

                for text, conf in zip(texts, scores):
                    if text:
                        clean_text = str(text).strip()
                        all_texts.append(clean_text)
                        total_conf += float(conf)
                        count += 1

            if not all_texts:
                return None, 0.0

            combined_text = "".join(all_texts).strip()

            # 1. allowlist
            combined_text = self._apply_allowlist(combined_text)

            # 2. нормализация
            combined_text = self._normalize_plate(combined_text)

            # 3. очистка
            final_text = self._clean_ocr_text(combined_text)

            logger.debug(f"OCR raw: {all_texts}")
            logger.debug(f"After allowlist: {combined_text}")

            avg_conf = total_conf / count if count > 0 else 0.0

            if visualize:
                save_paddleocr_visualization(
                    cropped_image,
                    result,
                    image_name,
                    avg_conf,
                    logger,
                    output_dir=self.output_vis_dir,
                )

            return final_text, avg_conf

        except Exception as e:
            logger.error(f"Ошибка PaddleOCR: {e}")
            return None, 0.0

    def _apply_allowlist(self, text: str) -> str:
        """Оставляет только разрешённые символы."""
        return "".join(c for c in text if c in self.allowlist)

    def _normalize_plate(self, text: str) -> str:
        text = text.upper()

        # частые ошибки OCR
        replacements = {
            "O": "0",
            "I": "1",
            "Z": "2",
            "B": "8",
            "S": "5",
            "L": "1",
            "l": "1",
            "i": "1",
        }

        corrected = ""
        for c in text:
            corrected += replacements.get(c, c)

        return corrected

    def _clean_ocr_text(self, text: str) -> Optional[str]:
        if not text:
            return None

        # Минимальная очистка: только нижний регистр + удаление пробелов в начале/конце
        cleaned = text.strip().lower()

        # Оставляем только если есть хотя бы одна цифра
        if not any(c.isdigit() for c in cleaned):
            return None

        # Убираем только самые очевидные мусорные символы
        cleaned = "".join(c for c in cleaned if c.isalnum() or c in " -_.")

        return cleaned.strip()
