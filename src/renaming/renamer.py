# src/renaming/renamer.py

from pathlib import Path
from typing import Optional


class FileRenamer:
    """
    Класс для генерации новых имён файлов.
    Сейчас: простой формат prefix_number[_suffix].ext
    Позже: будет поддержка группировки по EXIF и суффиксов _1, _2 и т.д.
    """

    def __init__(self, default_prefix: str = "SEFER"):
        self.default_prefix = default_prefix
        self.counter = {}  # для будущей группировки по номеру

    def generate_name(
        self,
        original_path: Path,
        number: str,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ) -> str:
        """
        Генерирует новое имя файла.

        Пример: SEFER_1234e_01.jpg
        """
        if prefix is None:
            prefix = self.default_prefix

        ext = original_path.suffix.lower()

        if suffix:
            new_name = f"{prefix}_{number}_{suffix}{ext}"
        else:
            new_name = f"{prefix}_{number}{ext}"

        return new_name

    def generate_name_with_counter(self, original_path: Path, number: str) -> str:
        """
        Заглушка для будущей версии с автоматическими суффиксами (_1, _2...).
        Сейчас просто использует generate_name.
        """
        # В будущем здесь будет логика:
        # - Группировка по EXIF времени
        # - Подсчёт дублей одного номера
        return self.generate_name(original_path, number)
