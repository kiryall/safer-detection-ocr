# utils/yaml_loader.py

import yaml
from pathlib import Path

def load_config(path: Path) -> dict:
    """Загружает конфигурацию из YAML файла.

    Args:
        path (Path): Путь к YAML файлу конфигурации.

    Returns:
        dict: Загруженная конфигурация.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)