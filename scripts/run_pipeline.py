# scripts/run_pipeline.py

import argparse
from src.pipeline.main_pipeline import run_pipeline
from configs.config import CONF_THRESHOLD, OCR_ACCEPT_THRESHOLD, YOLO_CONF_THRESHOLD, CONFIDANCE_LEVEL_THRESHOLD

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SEFER Vision Pipeline — обработка фотографий табличек",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Пример использования:\n"
               "  python -m scripts.run_pipeline --input_dir data/raw --prefix SEFER\n"
               "  python -m scripts.run_pipeline --conf 0.40 --ocr-conf 0.70 --visualize"
    )

    # === Пути ===
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw",
        help="Путь к папке с входными изображениями"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Путь к выходной папке (будут созданы success / low_confidence / very_low)"
    )
    
    # === Параметры ===
    parser.add_argument(
        "--prefix",
        type=str,
        default="SEFER",
        help="Префикс для новых имён файлов (например: SEFER, MDZ, AUTO)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Максимальное количество изображений для обработки (для тестов)"
    )

    # === Пороги уверенности ===
    parser.add_argument(
        "--conf",
        type=float,
        default=CONF_THRESHOLD,
        help="CONF_THRESHOLD — порог для статуса 'success'"
    )
    parser.add_argument(
        "--ocr-conf",
        type=float,
        default=OCR_ACCEPT_THRESHOLD,
        help="Минимальный confidence OCR для принятия результата"
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=YOLO_CONF_THRESHOLD,
        help="Порог уверенности YOLO детекции"
    )
    parser.add_argument(
        "--confidence-level-threshold",
        type=float,
        default=CONFIDANCE_LEVEL_THRESHOLD,
        help="Общий порог уверенности (det_conf * text_conf) для статуса 'success'"
    )

    # === Флаги ===
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Включить визуализации OCR и debug-кропы"
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Не перемешивать порядок обработки файлов"
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Очистить выходную папку перед запуском"
    )
    parser.add_argument(
        "--debug_save_crops",
        action="store_true",
        help="Сохраняет вырезанные таблички для дебага"
    )

    args = parser.parse_args()

    # Ensure output directory exists
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # Запуск пайплайна с указанными аргументами
    """Базовый запуск
    python -m scripts.run_pipeline --input_dir data/raw --prefix SEFER"""

    """С кастомными параметрами

    python -m scripts.run_pipeline `
    --input_dir data/raw `
    --output_dir output `
    --prefix test `
    --yolo-conf 0.1 `
    --ocr-conf 0.8 `
    --conf 0.40 `
    --confidence-level-threshold 0.8 `
    --limit 10 `
    --visualize `
    --clear-output `
    --debug_save_crops

    """

    print("=" * 95)
    print("🚀 SEFER VISION PIPELINE")
    print("=" * 95)
    print(f"📂 Входная папка     : {args.input_dir}")
    print(f"📁 Выходная папка    : {args.output_dir}")
    print(f"🏷️  Префикс          : {args.prefix}")
    print(f"🎯 YOLO conf         : {args.yolo_conf}")
    print(f"✅ CONF_THRESHOLD    : {args.conf}")
    print(f"🔤 OCR accept conf   : {args.ocr_conf}")
    print(f"📊 Confidence level  : {args.confidence_level_threshold}")
    if args.limit:
        print(f"⛔ Лимит файлов      : {args.limit}")
    print("-" * 95)
    
    results = run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prefix=args.prefix,
        limit=args.limit,
        yolo_conf=args.yolo_conf,
        conf_threshold=args.conf,
        ocr_conf_threshold=args.ocr_conf,
        confidence_level_threshold=args.confidence_level_threshold,
        visualize=args.visualize,
        shuffle=not args.no_shuffle,
        clear_output=args.clear_output,
        debug_save_crops=args.debug_save_crops,
        show_progress=True,
    )
