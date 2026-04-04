# src/ocr/ocr_engine.py

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import easyocr

from src.utils.logger import setup_logging

logger = setup_logging(log_file="ocr_engine.log", console=True, remove_file=True, logger_name="ocr_engine")

class OCREngine():
    """OCR движок для распознавания текста на табличках."""

    def __init__(self, gpu: bool = False):
        self.reader = easyocr.Reader(['en'], gpu=gpu)
        self.allowlist = '1234567890abcdeABCDElLiI'

    def recognize(self, cropped_image, image_name: str = "crop", visualize: bool = False) -> Tuple:
        """Распознаёт текст с защитой от неправильного типа данных.
        Args:
            cropped_image: Обрезанное изображение в формате PIL или None.
            image_name: Имя файла для сохранения визуализации.
            visualize: Флаг для сохранения визуализации.
        Returns:
            Кортеж (распознанный текст, confidence) или (None, 0.0) если распознавание не удалось.
        """
        if cropped_image is None:
            return None, 0.0

        # Защита: если случайно передали tuple
        if isinstance(cropped_image, tuple):
            logger.warning("В OCR передан tuple вместо Image!")
            cropped_image = cropped_image[0] if cropped_image else None

        if not isinstance(cropped_image, Image.Image):
            logger.error(f"OCR получил некорректный тип: {type(cropped_image)}")
            return None, 0.0

        try:
            if cropped_image.mode != "RGB":
                cropped_image = cropped_image.convert("RGB")

            img_np = np.asarray(cropped_image, dtype=np.uint8)

            if len(img_np.shape) == 2:           # grayscale
                img_np = np.stack([img_np] * 3, axis=-1)
            elif img_np.shape[-1] == 4:          # RGBA
                img_np = img_np[..., :3]

            results = self.reader.readtext(
                img_np,
                allowlist=self.allowlist,
                detail=1,
                paragraph=False,
                rotation_info=[0, 45, 90, 135, 180, 270],
                min_size=8,
                text_threshold=0.4,      # или даже 0.2–0.3
                low_text=0.3,
            )

            if not results:
                return None, 0.0

            all_texts = []
            total_conf = 0.0
            count = 0

            for bbox, text, prob in results:
                cleaned_char = ''.join(c for c in text if c.isdigit() or c.lower() in 'abcde')
                if cleaned_char:
                    all_texts.append(cleaned_char)
                    total_conf += float(prob)
                    count += 1

            if not all_texts:
                return None, 0.0

            # Объединяем все символы в один номер
            combined_text = ''.join(all_texts).lower()

            # Финальная очистка и валидация
            final_text = self._clean_ocr_text(combined_text)

            avg_conf = total_conf / count if count > 0 else 0.0

            if visualize:
                self._save_ocr_visualization(cropped_image, results, image_name, avg_conf)

            return final_text, avg_conf

        except Exception as e:
            logger.error(f"Ошибка OCR: {e}")
            return None, 0.0
        
    def _clean_ocr_text(self, text: str) -> Optional[str]:
        """Очистка и нормализация распознанного текста."""
        if not text:
            return None

        # Убираем всё кроме цифр и разрешённых букв
        cleaned = ''.join(c for c in text if c.isdigit() or c.lower() in 'abcde')

        # Приводим буквы к нижнему регистру
        cleaned = cleaned.lower()

        # Должна быть хотя бы одна цифра
        if not any(c.isdigit() for c in cleaned):
            return None

        return cleaned


    def _save_ocr_visualization(self, image: Image.Image, results, image_name: str, best_conf: float):
        """Сохраняет изображение с нарисованными боксами и текстом OCR."""
        try:
            viz_dir = Path("output/visualizations/ocr")
            viz_dir.mkdir(parents=True, exist_ok=True)

            img_draw = image.copy()
            draw = ImageDraw.Draw(img_draw)
            font = ImageFont.load_default()  # можно заменить на лучший шрифт позже

            for bbox, text, prob in results:
                # bbox в формате easyocr: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                points = [(int(p[0]), int(p[1])) for p in bbox]
                draw.polygon(points, outline="red", width=2)
                draw.text((points[0][0], points[0][1] - 15), 
                         f"{text} ({prob:.2f})", 
                         fill="red", font=font)

            conf_str = f"{best_conf:.3f}".replace(".", "_")
            save_path = viz_dir / f"ocr_{image_name}_conf{conf_str}.jpg"
            img_draw.save(save_path, quality=95)
            logger.debug(f"Визуализация OCR сохранена: {save_path}")

        except Exception as e:
            logger.warning(f"Не удалось сохранить визуализацию OCR: {e}")
        