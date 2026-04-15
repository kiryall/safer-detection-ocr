"""Компоненты для отображения метрик и статистики."""
import streamlit as st
import pandas as pd


def render_status_metrics(df: pd.DataFrame):
    """
    Отображение метрик статусов обработки.
    
    Args:
        df: DataFrame с результатами
    """
    if df.empty:
        st.info("Нет данных для отображения метрик")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        success_count = len(df[df["status"] == "success"])
        st.metric("✅ Успешно", success_count)
    
    with col2:
        low_conf_count = len(df[df["status"] == "low_confidence"])
        st.metric("⚠️ Сомнительно", low_conf_count)
    
    with col3:
        fail_count = len(df[df["status"] == "very_low_confidence"])
        st.metric("❌ Не распознано", fail_count)
    
    with col4:
        st.metric("📊 Всего", len(df))


def render_processing_time(seconds: float):
    """Отображение времени обработки."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    
    if minutes > 0:
        time_str = f"{minutes} мин {secs:.1f} сек"
    else:
        time_str = f"{secs:.1f} сек"
    
    st.success(f"✅ Обработка завершена за {time_str}!")
