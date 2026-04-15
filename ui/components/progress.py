"""Компонент прогресса с корректным обновлением через session_state."""
import streamlit as st


def init_progress_state():
    """Инициализация состояния прогресса."""
    if "progress_current" not in st.session_state:
        st.session_state.progress_current = 0
    if "progress_total" not in st.session_state:
        st.session_state.progress_total = 0
    if "progress_status" not in st.session_state:
        st.session_state.progress_status = ""
    if "progress_filename" not in st.session_state:
        st.session_state.progress_filename = ""


def update_progress(current: int, total: int, filename: str = ""):
    """Обновление прогресса через session_state."""
    st.session_state.progress_current = current
    st.session_state.progress_total = total
    st.session_state.progress_filename = filename
    st.session_state.progress_status = f"Обработка: {current}/{total}"


def reset_progress():
    """Сброс прогресса."""
    st.session_state.progress_current = 0
    st.session_state.progress_total = 0
    st.session_state.progress_status = ""
    st.session_state.progress_filename = ""


def render_progress_bar():
    """Отрисовка прогресс-бара и статуса."""
    init_progress_state()
    
    current = st.session_state.progress_current
    total = st.session_state.progress_total
    
    if total > 0:
        percent = min(int((current / total) * 100), 100)
    else:
        percent = 0
    
    # Прогресс-бар
    progress_bar = st.progress(percent / 100.0)
    
    # Статусная строка
    cols = st.columns([3, 1])
    with cols[0]:
        st.text(st.session_state.progress_status)
    with cols[1]:
        st.text(f"{percent}%")
    
    # Текущий файл
    if st.session_state.progress_filename:
        st.caption(f"🖼️ {st.session_state.progress_filename}")
    
    return progress_bar


def render_spinner(text: str = "Загрузка..."):
    """Отображение спиннера загрузки."""
    with st.spinner(text):
        yield
