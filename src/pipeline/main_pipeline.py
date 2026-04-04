# src/pipeline/main_pipeline.py

from pathlib import Path
from typing import List, Dict, Any, Optional

from src.pipeline.batch_processor import BatchProcessor
from src.reporting.report_generator import ReportGenerator
from src.utils.logger import setup_logging
from configs.config import RAW_DATA_DIR, OUTPUT_DIR

logger = setup_logging(log_file="main_pipeline.log", console=True, remove_file=True, logger_name="main_pipeline")


class SeferPipeline:
    """
    Главный класс пайплайна проекта
    """

    def __init__(self, prefix: str = "SEFER"):
        self.processor = BatchProcessor(prefix=prefix)
        self.report_generator = ReportGenerator(output_dir=OUTPUT_DIR)

    def run(self, 
            input_dir: Optional[Path] = None, 
            limit: Optional[int] = None,
            prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Запуск полного пайплайна.
        """
        if input_dir is None:
            input_dir = RAW_DATA_DIR

        if prefix:
            self.processor.renamer.default_prefix = prefix

        logger.info("=" * 60)
        logger.info("ЗАПУСК ПАЙПЛАЙНА SEFER VISION")
        logger.info(f"Входная папка : {input_dir}")
        logger.info(f"Выходная папка: {OUTPUT_DIR}")
        if limit:
            logger.info(f"Ограничение   : {limit} файлов")
        logger.info("=" * 60)

        # Основная обработка
        results = self.processor.process_folder(input_dir, limit=limit)

        # Сохранение отчёта
        self.report_generator.save_report(results)

        # Итоговая статистика
        total = len(results)
        success = sum(1 for r in results if r["status"] == "success")
        partial = sum(1 for r in results if r["status"] == "partial")

        logger.info("=" * 60)
        logger.info("ПАЙПЛАЙН ЗАВЕРШЁН")
        logger.info(f"Всего обработано файлов : {total}")
        logger.info(f"Успешно (с номером)     : {success}")
        logger.info(f"Частично                : {partial}")
        logger.info(f"Неудачно                : {total - success - partial}")
        logger.info("=" * 60)

        return results


def run_pipeline(input_dir: Optional[Path] = None, 
                 limit: Optional[int] = None, 
                 prefix: str = "SEFER"):
    """Запуск пайплайна с параметрами."""
    pipeline = SeferPipeline(prefix=prefix)
    return pipeline.run(input_dir=input_dir, limit=limit)


if __name__ == "__main__":
    # Для быстрого теста
    run_pipeline(limit=20)   # обработает максимум 20 фото