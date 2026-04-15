# src/utils/ocr_utils.py

import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from configs.config import OUTPUT_VIS_DIR


def save_ocr_visualization(
    image: Image.Image,
    detections: list,
    image_name: str,
    best_conf: float,
    logger: logging.Logger,
    output_dir: Path | str | None = None,
):
    """Сохраняет изображение с нарисованными боксами и текстом OCR."""

    logger.info(
        f"Создание визуализации OCR для {image_name} с confidence {best_conf:.2f}"
    )

    vis_dir = Path(output_dir) if output_dir else OUTPUT_VIS_DIR
    vis_dir.mkdir(parents=True, exist_ok=True)

    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)

    font = ImageFont.load_default()

    for det in detections:
        box = det["box"]
        char = det["char"]
        conf = det["conf"]

        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        text_str = f"{char} ({conf:.2f})"
        draw.text(
            (x1, y1 - 15),
            text_str,
            fill="red",
            font=font,
        )

    vis_path = vis_dir / f"{image_name}_ocr_vis.png"
    image_copy.save(vis_path)
    logger.info(f"Визуализация сохранена: {vis_path}")


def save_paddleocr_visualization(
    image: Image.Image,
    results,
    image_name: str,
    best_conf: float,
    logger: logging.Logger,
    output_dir: Path | str | None = None,
):
    """Визуализация OCR (PaddleOCR predict API)."""

    logger.info(
        f"Создание визуализации OCR для {image_name} с confidence {best_conf:.2f}"
    )

    vis_dir = Path(output_dir) if output_dir else OUTPUT_VIS_DIR
    vis_dir.mkdir(parents=True, exist_ok=True)

    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)

    font = ImageFont.load_default()

    for res in results:
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        boxes = res.get("dt_polys", [])

        for text, conf, bbox in zip(texts, scores, boxes):
            if not text:
                continue

            try:
                # bbox — это 4 точки [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                points = [(int(x), int(y)) for x, y in bbox]

                # рисуем полигон
                draw.polygon(points, outline="red")

                # берем верхнюю левую точку
                x, y = points[0]

                text_str = f"{text} ({conf:.2f})"
                draw.rectangle([x, y - 18, x + 120, y], fill="black")
                draw.text((x, y - 18), text_str, fill="white", font=font)

            except Exception as e:
                logger.warning(f"Ошибка при отрисовке бокса: {e}")

    vis_path = vis_dir / f"{image_name}_ocr_vis.png"
    image_copy.save(vis_path)

    logger.info(f"Визуализация сохранена: {vis_path}")
