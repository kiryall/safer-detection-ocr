# src/pipeline/main_pipeline.py

from pathlib import Path
import shutil
from typing import List, Dict, Any, Optional, Callable

from tqdm import tqdm

from src.pipeline.batch_processor import BatchProcessor
from src.reporting.report_generator import ReportGenerator
from src.utils.logger import setup_logging
from configs.config import (
    RAW_DATA_DIR,
    OUTPUT_DIR,
    SUPPORTED_IMAGE_FORMATS,
    LOGGING_CONSOLE,
)

logger = setup_logging(
    log_file="main_pipeline.log",
    console=LOGGING_CONSOLE,
    remove_file=True,
    logger_name="main_pipeline",
)


class SeferPipeline:
    """
    Главный класс пайплайна проекта
    """

    def __init__(self, prefix: str = "SEFER", visualize: bool = False):
        self.prefix = prefix
        self.visualize = visualize
        self.report_generator = ReportGenerator(output_dir=None)

    def run(
        self,
        input_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        limit: Optional[int] = None,
        prefix: Optional[str] = None,
        yolo_conf: float = 0.50,
        conf_threshold: float = None,
        ocr_conf_threshold: float = 0.0,
        confidence_level_threshold: float = None,
        visualize: bool = False,
        shuffle: bool = True,
        clear_output: bool = False,
        show_progress: bool = True,
        debug_save_crops: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,   # ← Новый параметр
    ) -> List[Dict[str, Any]]:
        """
        Запуск полного пайплайна с прогресс-баром.
        """
        input_dir = Path(input_dir) if input_dir else RAW_DATA_DIR
        output_dir = Path(output_dir) if output_dir else OUTPUT_DIR

        if prefix:
            self.prefix = prefix

        logger.info("=" * 80)
        logger.info("ЗАПУСК ПАЙПЛАЙНА SEFER VISION")
        logger.info(f"Входная папка : {input_dir}")
        logger.info(f"Выходная папка: {output_dir}")
        logger.info(f"Префикс       : {self.prefix}")
        if limit:
            logger.info(f"Лимит         : {limit} файлов")
        logger.info("=" * 80)

        # === Очистка выходной папки ===
        if clear_output and output_dir.exists():
            logger.info("Очистка выходной папки...")
            for folder in [
                "success",
                "low_confidence",
                "very_low",
                "crops",
                    "visualizations",
                "metadata",
            ]:
                folder_path = output_dir / folder
                if folder_path.exists():
                    shutil.rmtree(folder_path, ignore_errors=True)
            logger.info("Выходная папка успешно очищена.")

        # === Создание процессора ===
        processor = BatchProcessor(
            prefix=self.prefix,
            visualize=visualize,
            base_output_dir=output_dir,
            conf_threshold=conf_threshold,
            yolo_conf_threshold=yolo_conf,
            ocr_conf_threshold=ocr_conf_threshold,
            confidence_level_threshold=confidence_level_threshold,
            debug_save_crops=debug_save_crops
        )

        # === Подготовка списка файлов ===
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

        # Применяем лимит
        if limit:
            if shuffle:
                import random
                from configs.config import SEED

                rnd = random.Random(SEED)
                image_paths = rnd.sample(image_paths, min(limit, len(image_paths)))
            else:
                image_paths = image_paths[:limit]

        total_files = len(image_paths)
        logger.info(f"Найдено {total_files} изображений для обработки.")

        # === Основная обработка с прогресс-баром ===
        results = []
        progress_bar = None

        if show_progress and not progress_callback:   # для CLI используем tqdm
            progress_bar = tqdm(total=total_files, desc="Обработка фотографий", unit="фото")

        try:
            for i, image_path in enumerate(image_paths, 1):
                result = processor.process_single_image(image_path)
                results.append(result)

                # Обновляем прогресс
                if progress_callback:
                    progress_callback(i, total_files)          # ← Для Streamlit
                elif progress_bar:
                    progress_bar.update(1)

                # Логирование каждые 20 файлов
                if i % 20 == 0 or i == total_files:
                    success = sum(1 for r in results if r.get("status") == "success")
                    low = sum(1 for r in results if r.get("status") == "low_confidence")
                    logger.info(
                        f"Обработано {i}/{total_files} | Успешно: {success} | Сомнительно: {low}"
                    )

        finally:
            if progress_bar:
                progress_bar.close()

        # === Сохранение отчёта ===
        self.report_generator.output_dir = output_dir
        self.report_generator.save_report(results)

        # === Итоговая статистика ===
        total = len(results)
        success = sum(1 for r in results if r["status"] == "success")
        low_conf = sum(1 for r in results if r["status"] == "low_confidence")
        failed = total - success - low_conf

        logger.info("=" * 80)
        logger.info("ПАЙПЛАЙН ЗАВЕРШЁН")
        logger.info(f"Всего обработано     : {total}")
        logger.info(f"Успешно              : {success}")
        logger.info(f"С сомнением          : {low_conf}")
        logger.info(f"Неудачно             : {failed}")
        logger.info("=" * 80)

        return results


def run_pipeline(
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    limit: Optional[int] = None,
    prefix: str = "SEFER",
    yolo_conf: float = 0.50,
    conf_threshold: float = None,
    ocr_conf_threshold: float = 0.0,
    confidence_level_threshold: float = None,
    visualize: bool = False,
    shuffle: bool = True,
    clear_output: bool = False,
    show_progress: bool = True,
    debug_save_crops: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None
):
    """обёртка для запуска."""
    pipeline = SeferPipeline(prefix=prefix, visualize=visualize)
    return pipeline.run(
        input_dir=input_dir,
        output_dir=output_dir,
        limit=limit,
        prefix=prefix,
        yolo_conf=yolo_conf,
        conf_threshold=conf_threshold,
        ocr_conf_threshold=ocr_conf_threshold,
        confidence_level_threshold=confidence_level_threshold,
        visualize=visualize,
        shuffle=shuffle,
        clear_output=clear_output,
        show_progress=show_progress,
        debug_save_crops=debug_save_crops,
        progress_callback=progress_callback,
    )


if __name__ == "__main__":
    # Для быстрого теста
    run_pipeline(limit=10, visualize=False)
