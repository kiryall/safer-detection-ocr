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
        self.counter = {}
        
    def _validate_number(self, number: str = None):
        """
        Приводит номер фото к виду SEFER_1234e_01.jpg
        Возвращает строку длиной минимум 4 символа, дополненную нулями слева.
        """
        if number is None:
            number = ""
        return number.zfill(4)


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

        # валидация номера
        number = self._validate_number(number)

        if suffix:
            new_name = f"{prefix}{number}_{suffix}{ext}"
        else:
            new_name = f"{prefix}{number}{ext}"

        return new_name
    

    def generate_name_with_counter(
        self, original_path: Path, number: str, low_confidence: bool = False
    ) -> str:
        """
        Генерирует имя файла и добавляет суффиксы _1, _2... для повторяющихся имён.
        Первое совпадение оставляется без суффикса.
        """
        suffix = "lc" if low_confidence else None
        base_name = self.generate_name(original_path, number, suffix=suffix)

        count = self.counter.get(base_name, 0)
        if count == 0:
            self.counter[base_name] = 1
            return base_name
        
        self.counter[base_name] = count + 1
        root = Path(base_name).stem
        ext = Path(base_name).suffix
        return f"{root}_{count}{ext}"
