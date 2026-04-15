"""Компоненты для отображения результатов обработки."""
import streamlit as st
import pandas as pd
from pathlib import Path


def render_best_result(df: pd.DataFrame):
    """
    Отображение лучшего результата.
    
    Args:
        df: DataFrame с результатами
    """
    success_df = df[df["status"] == "success"]
    
    if success_df.empty:
        st.info("Нет успешных результатов")
        return
    
    # Получаем лучший результат по detection_confidence
    best = success_df.loc[success_df["detection_confidence"].idxmax()]
    
    st.markdown("### ✅ Самый удачный результат")
    st.write(f"**Файл:** `{best['filename']}`")
    
    cols = st.columns(2)
    with cols[0]:
        det_conf = best.get('detection_confidence', 0)
        st.write(f"🎯 Детекция: **{det_conf:.3f}**")
    with cols[1]:
        text_conf = best.get('text_confidence', 0)
        st.write(f"📝 OCR: **{text_conf:.3f}**")
    
    clean_text = best.get('clean_text', best.get('recognized_text', '—'))
    st.write(f"**Текст:** `{clean_text}`")
    
    # Попытаемся отобразить финальное изображение
    final_path = best.get("final_path", best.get("crop_path", ""))
    if final_path and Path(final_path).exists():
        st.image(str(final_path), use_container_width=True)


def render_worst_result(df: pd.DataFrame):
    """
    Отображение худшего результата.
    
    Args:
        df: DataFrame с результатами
    """
    if df.empty:
        st.info("Нет данных")
        return
    
    worst = df.loc[df["detection_confidence"].idxmin()]
    
    st.markdown("### ❌ Самый сложный случай")
    st.write(f"**Файл:** `{worst['filename']}`")
    st.write(f"📊 Статус: **{worst['status']}**")
    
    det_conf = worst.get('detection_confidence', 0)
    st.write(f"🎯 Детекция: **{det_conf:.3f}**")
    
    recognized = worst.get('recognized_text', worst.get('clean_text', '—'))
    st.write(f"📝 Текст: `{recognized}`")
    
    # Попытаемся отобразить оригинальное изображение
    orig_path = worst.get("original_path", worst.get("image_path", ""))
    if orig_path and Path(orig_path).exists():
        st.image(str(orig_path), use_container_width=True)


def render_results_table(df: pd.DataFrame):
    """
    Отображение таблицы всех результатов.
    
    Args:
        df: DataFrame с результатами
    """
    st.subheader("📋 Все результаты (от самых подозрительных сверху)")
    
    if df.empty:
        st.info("Нет данных для отображения")
        return

    # названия колонок
    column_rename = {
        "filename": "Имя_файла",
        "original_path": "Путь_к_оригиналу",
        "detection_confidence": "Уверенность_детекции",
        "recognized_text": "Распознанный_текст",
        "text_confidence": "Уверенность_OCR",
        "clean_text": "Чистый_номер",
        "final_name": "Новое_имя_файла",
        "final_path": "Путь_к_финальному_файлу",
        "status": "Статус",
        "message": "Сообщение",
        'confidence_level': 'Общий_уровень_уверенности'
    }

    df = df.rename(columns=column_rename)

    # Сортировка по confidence
    df_sorted = df.sort_values(by='Общий_уровень_уверенности', ascending=True)
    
    # Выбор колонок для отображения (в порядке приоритета)
    possible_cols = [
        "Имя_файла", 
        "Статус", 
        "Уверенность_детекции",
        "Уверенность_OCR", 
        "Распознанный_текст",
        "Новое_имя_файла", 
        "Сообщение",
        'Общий_уровень_уверенности'
    ]
    
    # Фильтруем только существующие колонки
    existing_cols = [c for c in possible_cols if c in df_sorted.columns]
    
    st.dataframe(
        df_sorted[existing_cols],
        use_container_width=True,
        hide_index=True,
    )
    
    # Кнопка скачивания
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Скачать полный CSV-отчёт",
        data=csv,
        file_name="processing_results.csv",
        mime="text/csv",
        use_container_width=True
    )
