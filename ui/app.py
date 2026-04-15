"""
SEFER Vision — Приложение для обработки фотографий табличек.
"""
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import threading
import time
import queue

# Добавляем корень проекта в PATH (делаем это ДО других импортов)
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Импорты компонентов
from components.sidebar import render_sidebar, PipelineConfig
from components.progress import (
    init_progress_state, 
    render_progress_bar, 
    reset_progress,
    update_progress
)
from components.metrics import render_status_metrics, render_processing_time
from components.results_display import (
    render_best_result, 
    render_worst_result, 
    render_results_table
)
from pipeline_wrapper import PipelineRunner

# Импорт конфигов (после добавления пути)
from configs.config import CONF_THRESHOLD, OCR_ACCEPT_THRESHOLD, YOLO_CONF_THRESHOLD, CONFIDANCE_LEVEL_THRESHOLD


def run_pipeline_in_thread(config, progress_queue):
    """Запуск пайплайна в отдельном потоке."""
    runner = PipelineRunner(config, progress_queue=progress_queue)
    results, processing_time = runner.run()
    progress_queue.put(('done', results, processing_time))


# ====================== КОНФИГУРАЦИЯ СТРАНИЦЫ ======================
st.set_page_config(
    page_title="SEFER Vision",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 SEFER Vision — Обработка фотографий табличек")


# ====================== БОКОВАЯ ПАНЕЛЬ ======================
defaults = {
    "prefix": "SEFER",
    "input_dir": "data/raw",
    "output_dir": "output",
    "yolo_conf": YOLO_CONF_THRESHOLD,
    "conf_threshold": CONF_THRESHOLD,
    "ocr_conf_threshold": OCR_ACCEPT_THRESHOLD,
    "confidence_level_threshold": CONFIDANCE_LEVEL_THRESHOLD,
}

config = render_sidebar(defaults)


# ====================== ОСНОВНОЙ КОНТЕНТ ======================
def validate_inputs(config: PipelineConfig) -> bool:
    """Валидация входных данных."""
    input_path = Path(config.input_dir)
    
    if not input_path.exists():
        st.error(f"❌ Папка не найдена: {input_path}")
        return False
    
    if not input_path.is_dir():
        st.error(f"❌ Указанный путь не является папкой: {input_path}")
        return False
    
    return True


def prepare_config_dict(config: PipelineConfig) -> dict:
    """Подготовка конфигурации для пайплайна."""
    return {
        "input_dir": config.input_dir,
        "output_dir": config.output_dir,
        "prefix": config.prefix,
        "limit": config.limit,
        "yolo_conf": config.yolo_conf,
        "conf_threshold": config.conf_threshold,
        "ocr_conf_threshold": config.ocr_conf_threshold,
        "confidence_level_threshold": config.confidence_level_threshold,
        "visualize": config.visualize,
        "shuffle": config.shuffle,
        "clear_output": config.clear_output,
        "debug_save_crops": config.debug_save_crops,
    }


# ====================== КНОПКА ЗАПУСКА ======================
button_clicked = st.button(
    "🚀 Запустить обработку", 
    type="primary", 
    use_container_width=True
)

# Инициализация состояния прогресса
init_progress_state()

# Если нажата кнопка или идет обработка
if button_clicked:
    if not validate_inputs(config):
        st.stop()
    
    # Сброс прогресса
    reset_progress()
    st.session_state.processing = True
    
    # Запуск в отдельном потоке
    pipeline_config = prepare_config_dict(config)
    progress_queue = queue.Queue()
    thread = threading.Thread(target=run_pipeline_in_thread, args=(pipeline_config, progress_queue))
    st.session_state.runner_thread = thread
    st.session_state.progress_queue = progress_queue
    thread.start()
    
    # Принудительный rerun для отображения прогресса
    st.rerun()

# Отображение прогресса и обработка
if st.session_state.get("processing", False):
    thread = st.session_state.get("runner_thread")
    progress_queue = st.session_state.get("progress_queue")
    
    # Обновляем прогресс из очереди
    if progress_queue:
        while not progress_queue.empty():
            msg = progress_queue.get()
            if msg[0] == 'done':
                st.session_state.results = msg[1]
                st.session_state.processing_time = msg[2]
            else:
                current, total, filename = msg
                update_progress(current, total, filename)
    
    if thread and thread.is_alive():
        # Показываем прогресс-бар
        st.subheader("⏳ Прогресс обработки")
        render_progress_bar()
        
        # Небольшая задержка и rerun для обновления
        time.sleep(0.1)
        st.rerun()
    else:
        # Обработка завершена
        st.session_state.processing = False
        
        # Финальное обновление прогресса
        update_progress(100, 100, "Завершено!")
        
        # Успешное завершение
        st.success("✅ Обработка завершена!")
        st.rerun()

# ====================== РЕЗУЛЬТАТЫ ======================
if "results" in st.session_state and st.session_state.results:
    results = st.session_state.results
    processing_time = st.session_state.get("processing_time", 0)
    df = pd.DataFrame(results)
    
    # Статистика
    st.divider()
    render_processing_time(processing_time)
    render_status_metrics(df)
    
    # Примеры результатов
    st.subheader("🖼️ Примеры результатов")
    col_left, col_right = st.columns(2)
    
    with col_left:
        render_best_result(df)
    
    with col_right:
        render_worst_result(df)
    
    # Таблица результатов
    st.divider()
    render_results_table(df)