# src/postprocessing/text_cleaner.py

import re
from typing import Optional


def clean_validate_text(text: Optional[str]) -> Optional[str]:
    """
    Финальная очистка и валидация номера таблички.
    
    Правила:
    - Только цифры + одна буква a-e в конце
    - Убираем стрелки и лишние символы
    - Нормализация формата
    """
    if not text:
            return None

    # Убираем всё лишнее
    cleaned = ''.join(c for c in text if c.isdigit() or c.lower() in 'abcde')
    cleaned = cleaned.lower()

    # Должна быть хотя бы одна цифра
    if not any(c.isdigit() for c in cleaned):
        return None

    # Если есть буква — она должна быть только в конце
    match = re.match(r'^(\d+)([a-e])?$', cleaned)

    if match:
        return cleaned
    else:
        # Если буква не в конце — перемещаем её в конец
        digits = ''.join(c for c in cleaned if c.isdigit())
        letter = ''.join(c for c in cleaned if c in 'abcde')
        if letter:
            return digits + letter[0]
        return digits