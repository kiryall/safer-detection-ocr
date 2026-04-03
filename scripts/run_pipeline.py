# scripts/run_pipeline.py

import argparse
from src.pipeline.main_pipeline import run_pipeline
from configs.config import RAW_DATA_DIR

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Запуск пайплайна обработки изображений.")
    argparser.add_argument("--input_dir", type=str, default=str(RAW_DATA_DIR), help="Путь к папке с входными изображениями.")
    argparser.add_argument("--limit", type=int, default=10, help="Максимальное количество изображений для обработки (для тестирования).")
    args = argparser.parse_args()

    # Запуск пайплайна с указанными аргументами
    run_pipeline(input_dir=args.input_dir, limit=args.limit)