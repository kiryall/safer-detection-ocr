"""Вспомогательные функции для UI приложения."""
import sys
from pathlib import Path
import pandas as pd


def add_project_to_path():
    """Добавляет корень проекта в sys.path."""
    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def validate_input_directory(input_dir: str) -> tuple[bool, str]:
    """
    Валидирует входную директорию.
    
    Args:
        input_dir: путь к директории
    
    Returns:
        tuple: (is_valid, error_message)
    """
    path = Path(input_dir)
    
    if not path.exists():
        return False, f"Папка не найдена: {path}"
    
    if not path.is_dir():
        return False, f"Путь не является папкой: {path}"
    
    # Проверяем, есть ли файлы изображений
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    images = [f for f in path.rglob('*') if f.suffix.lower() in image_extensions]
    
    if not images:
        return False, f"В папке нет изображений: {path}"
    
    return True, ""


def load_results_csv(csv_path: str) -> pd.DataFrame:
    """
    Загружает результаты из CSV.
    
    Args:
        csv_path: путь к CSV файлу
    
    Returns:
        pd.DataFrame: таблица результатов
    """
    path = Path(csv_path)
    if path.exists():
        try:
            return pd.read_csv(str(path))
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def format_confidence(value: float, decimals: int = 3) -> str:
    """
    Форматирует значение confidence.
    
    Args:
        value: значение от 0 до 1
        decimals: количество знаков после запятой
    
    Returns:
        str: отформатированное значение с процентом
    """
    percent = value * 100
    return f"{value:.{decimals}f} ({percent:.1f}%)"


def get_status_emoji(status: str) -> str:
    """
    Возвращает эмодзи для статуса.
    
    Args:
        status: статус обработки
    
    Returns:
        str: эмодзи и описание
    """
    status_map = {
        "success": "✅ Успешно",
        "low_confidence": "⚠️ Сомнительно",
        "very_low_confidence": "❌ Не распознано",
        "error": "🔴 Ошибка",
    }
    return status_map.get(status, "❓ Неизвестно")
