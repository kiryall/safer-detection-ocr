# src/pipeline/main_pipeline.py

from pathlib import Path
from src.pipeline.batch_processor import BatchProcessor
from src.utils.logger import setup_logging

logger = setup_logging(log_file="main_pipeline.log", remove_file=True, console=True, logger_name="main_pipeline")


def run_pipeline(input_dir: Path = None, limit: int = None):
    """Удобная точка входа для запуска пайплайна."""
    if input_dir is None:
        from configs.config import RAW_DATA_DIR
        input_dir = RAW_DATA_DIR

    logger.info(f"Запуск пайплайна. Входная папка: {input_dir}")

    processor = BatchProcessor()
    results = processor.process_folder(input_dir, limit=limit)

    logger.info(f"Пайплайн завершён. Обработано: {len(results)} файлов.")
    
    # Здесь позже будет вызов отчёта
    return results


if __name__ == "__main__":
    run_pipeline(limit=10)   # для быстрого теста