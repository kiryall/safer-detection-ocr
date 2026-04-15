"""Обертка для запуска пайплайна с прогрессом."""
import sys
from pathlib import Path
import time
from typing import Callable, Optional

import streamlit as st
from src.pipeline.main_pipeline import run_pipeline
from components.progress import update_progress

# Добавляем корень проекта в PATH
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


class PipelineRunner:
    """Класс для запуска пайплайна с отслеживанием прогресса."""
    
    def __init__(self, config: dict, progress_callback: Optional[Callable] = None):
        """
        Инициализация runner'а.
        
        Args:
            config: словарь с конфигурацией пайплайна
        """
        self.config = config
        self.results = []
        self.processing_time = 0.0
        self.external_progress_callback = progress_callback
    
    def _create_callback(self) -> Callable:
        """Создание callback-функции для обновления прогресса."""
        if self.external_progress_callback:
            return self.external_progress_callback
        def callback(current: int, total: int):
            # Обновляем session_state для отображения в UI
            update_progress(current, total, f"Изображение {current} из {total}")
            # Небольшая задержка для обновления UI
            time.sleep(0.01)
        return callback
    
    def run(self) -> tuple[list, float]:
        """
        Запуск пайплайна.
        
        Returns:
            tuple: (результаты, время выполнения в секундах)
        
        Raises:
            Exception: если произошла ошибка при обработке
        """
        start_time = time.time()
        
        try:
            # Подготовка callback
            progress_callback = self._create_callback()
            
            # Запуск пайплайна с передачей callback
            self.results = run_pipeline(
                input_dir=self.config["input_dir"],
                output_dir=self.config["output_dir"],
                prefix=self.config["prefix"],
                limit=self.config["limit"],
                yolo_conf=self.config["yolo_conf"],
                conf_threshold=self.config["conf_threshold"],
                ocr_conf_threshold=self.config["ocr_conf_threshold"],
                confidence_level_threshold=self.config["confidence_level_threshold"],
                visualize=self.config["visualize"],
                shuffle=self.config["shuffle"],
                clear_output=self.config["clear_output"],
                debug_save_crops=self.config["debug_save_crops"],
                show_progress=False,  # отключаем tqdm, используем свой callback
                progress_callback=progress_callback,
            )
            
        except Exception as e:
            st.error(f"❌ Ошибка во время обработки: {e}")
            raise
        
        self.processing_time = time.time() - start_time
        return self.results, self.processing_time
