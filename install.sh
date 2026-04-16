#!/bin/bash
echo "Создание виртуального окружения..."
python3 -m venv venv
echo "Активация виртуального окружения..."
source venv/bin/activate
echo "Установка зависимостей..."
pip install -e .
echo "Установка завершена. Для активации окружения в будущем используйте: source venv/bin/activate"