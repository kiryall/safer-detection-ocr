"""Боковая панель с настройками приложения."""
import streamlit as st
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Конфигурация пайплайна обработки."""
    prefix: str
    input_dir: str
    output_dir: str
    limit: int | None
    yolo_conf: float
    conf_threshold: float
    ocr_conf_threshold: float
    confidence_level_threshold: float
    visualize: bool
    debug_save_crops: bool
    shuffle: bool
    clear_output: bool


def render_sidebar(defaults: dict) -> PipelineConfig:
    """
    Отрисовка боковой панели с настройками.
    
    Args:
        defaults: словарь с значениями по умолчанию
    
    Returns:
        PipelineConfig: объект с настройками
    """
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # Префикс файлов
        prefix = st.text_input(
            "Префикс имён файлов", 
            value=defaults.get("prefix", "SEFER")
        )
        
        # Папки ввода/вывода
        col1, col2 = st.columns(2)
        with col1:
            input_dir = st.text_input(
                "📂 Папка с фото", 
                value=defaults.get("input_dir", "data/raw")
            )
        with col2:
            output_dir = st.text_input(
                "📁 Выходная папка", 
                value=defaults.get("output_dir", "output")
            )
        
        # Лимит изображений
        limit_raw = st.number_input(
            "Лимит изображений (0 = все)", 
            value=0, 
            min_value=0, 
            step=10
        )
        limit = None if limit_raw == 0 else int(limit_raw)
        
        # Пороги уверенности
        st.subheader("Пороги уверенности")
        
        yolo_conf = st.slider(
            "Порог детекции таблички YOLO моделью",
            min_value=0.05,
            max_value=0.99,
            value=defaults.get("yolo_conf", 0.5),
            step=0.01
        )
        
        conf_threshold = st.slider(
            "Принимаемый порог детекции",
            min_value=0.05,
            max_value=0.99,
            value=defaults.get("conf_threshold", 0.5),
            step=0.01,
            help='Ниже данного порога детекция считается неудачной'
        )
        
        ocr_conf_threshold = st.slider(
            "Порог распознавания текста OCR",
            min_value=0.05,
            max_value=0.99,
            value=defaults.get("ocr_conf_threshold", 0.5),
            step=0.01
        )
        
        confidence_level_threshold = st.slider(
            "Общий порог уверенности",
            min_value=0.05,
            max_value=0.99,
            value=defaults.get("confidence_level_threshold", 0.5),
            step=0.01,
            help="Произведение уверенности детекции YOLO и OCR для статуса 'success'"
        )
        
        # Дополнительные опции
        st.subheader("Дополнительные опции")
        
        visualize = st.checkbox("Визуализация OCR", value=False)
        debug_save_crops = st.checkbox("Сохранять кропы для дебага", value=False)
        no_shuffle = st.checkbox("Не перемешивать порядок файлов", value=False)
        clear_output = st.checkbox(
            "🗑️ Очистить выходную папку перед запуском", 
            value=False
        )
        
        # Валидация путей
        input_path = Path(input_dir)
        if not input_path.exists():
            st.warning(f"⚠️ Папка не найдена: {input_path}")
        
        return PipelineConfig(
            prefix=prefix,
            input_dir=input_dir,
            output_dir=output_dir,
            limit=limit,
            yolo_conf=yolo_conf,
            conf_threshold=conf_threshold,
            ocr_conf_threshold=ocr_conf_threshold,
            confidence_level_threshold=confidence_level_threshold,
            visualize=visualize,
            debug_save_crops=debug_save_crops,
            shuffle=not no_shuffle,
            clear_output=clear_output
        )
