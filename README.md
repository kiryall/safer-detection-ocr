# SAFER: Детекция и OCR 🚀
> Инструмент для автоматического обнаружения объектов и распознавания текста на изображениях в полевых условиях. Ключевой результат: Точность распознавания до 95% на тестовых данных экспедиций.

[![Stars](https://img.shields.io/github/stars/kiryall/safer-detection-ocr)](https://github.com/kiryall/safer-detection-ocr/stargazers) [![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](https://www.python.org/)

## ✨ Возможности
- 🔍 **Автоматическая детекция объектов**: Использование YOLO для точного обнаружения элементов на изображениях.
- 📝 **Распознавание текста**: Интеграция YOLO для извлечения текста из обнаруженных областей.
- 🖥️ **Дружелюбный интерфейс**: Streamlit-приложение для удобного взаимодействия без кода.
- 📊 **Пакетная обработка**: Обработка больших наборов изображений.
- 🧹 **Постобработка**: Очистка и валидация текста для повышения точности.
- 📈 **Отчеты и визуализации**: Генерация CSV-отчетов и изображений с аннотациями.
- ⚙️ **Гибкая конфигурация**: Настраиваемые пороги уверенности и модели.
- 🌍 **Для экспедиций**: Оптимизировано для использования в полевых условиях с минимальными ресурсами.

## 📸 Скриншоты / Демо
![Интерфейс приложения]([img\interface.png](https://github.com/kiryall/safer-detection-ocr/raw/master/img/interface.png)) 
*Главный экран Streamlit-приложения для загрузки и обработки изображений.*

![Результат детекции и OCR]([img\demo.png](https://github.com/kiryall/safer-detection-ocr/raw/master/img/demo.png))  
*Пример обнаружения объектов и распознанного текста.*

> 💡 **Совет для экспедиций**: Снимайте фото при хорошем освещении и избегайте размытия для лучших результатов.

## 🚀 Быстрый старт
### Предварительные требования
- Python 3.11+
- CUDA-compatible GPU (рекомендуется для ускорения YOLO)

### Установка
#### Одной командой (с uv)
```bash
uv pip install -e .
```

#### Поэтапно
1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/kiryall/safer-detection-ocr.git
   cd safer-detection-ocr
   ```
2. Установите зависимости:
   ```bash
   pip install -e .
   ```
   Или с uv:
   ```bash
   uv sync
   ```

### Запуск
#### CLI
- Базовый запуск
```bash
python -m scripts.run_pipeline --input_dir data/raw --prefix SEFER
```
- С кастомными параметрами
```bash
python -m scripts.run_pipeline `
    --input_dir data/raw `
    --output_dir output `
    --prefix test `
    --yolo-conf 0.1 `
    --ocr-conf 0.8 `
    --conf 0.40 `
    --confidence-level-threshold 0.8 `
    --limit 10 `
    --visualize `
    --clear-output `
    --debug_save_crops
```

#### GUI
```bash
streamlit run ui/app.py
```
Откройте браузер по адресу http://localhost:8501.

## 📋 Как использовать
1. **Загрузка изображений**: укажите папку с фото для обработки, укажите выходную папку.
2. **Настройки**: Настройте пороги уверенности в config.py (например, CONF_THRESHOLD = 0.2).
3. **Результаты**: Просмотрите отчеты в output/results.csv и визуализации.

> ⚠️ **Предупреждение**: Для полевых условий убедитесь, что изображения в формате JPG/PNG и размером не более 10MB на файл.

## 🧠 Как это работает
- **Архитектура**: Модульная структура с разделением на детекцию, OCR и постобработку.
- **Pipeline**:
  - Загрузка и предобработка изображений.
  - Детекция объектов с YOLO.
  - Кроппинг обнаруженных областей.
  - Распознавание текста с OCR.
  - Валидация и очистка результатов.
  - Генерация отчетов.
- **Модели**: YOLO26s для детекции, предобученная YOLO для текста.
- **Постобработка**: Фильтрация по уверенности, очистка символов.

## 🛠️ Технологический стек
- **Язык**: Python 3.11+
- **Основные библиотеки**: Ultralytics (YOLO), Supervision, Pandas, Scikit-learn
- **GUI**: Streamlit
- **Модели**: Ultralytics (YOLO)

## 📁 Структура проекта
```
safer/
├── pyproject.toml    # Конфигурация проекта и зависимости
├── configs/          # Конфигурационные файлы
├── data/             # Данные и датасеты
├── logs/             # Логи
├── models/           # Предобученные модели
├── notebooks/        # Jupyter ноутбуки для анализа
├── output/           # Результаты обработки
├── scripts/          # Скрипты для запуска
├── src/              # Исходный код
│   ├── detection/    # Детекция с YOLO
│   ├── ocr/          # Распознавание текста
│   ├── pipeline/     # Основной пайплайн
│   └── ...
├── tests/            # Тесты
└── ui/               # Интерфейс Streamlit
```

## ⚙️ Конфигурация
Пример `configs/config.py`:
```python
CONF_THRESHOLD = 0.2  # Порог уверенности детекции
OCR_ACCEPT_THRESHOLD = 0.8  # Порог для OCR
ALLOWLIST = "0123456789"  # Разрешенные символы
```
Описание параметров:
- `CONF_THRESHOLD`: Минимальная уверенность для детекции (0-1).
- `OCR_ACCEPT_THRESHOLD`: Порог принятия OCR-результата.
- `ALLOWLIST`: Символы для распознавания (для цифр).

## 🔬 Результаты и бенчмарки
[📊 View Metrics Notebook](notebooks\calculate_metrics.ipynb)

- Accuracy = 0.79
- ROC-AUC  = 0.8058


## 🏗️ Разработка и вклад
- **Сборка из исходников**: `uv build`
- **Обучение модели**: Запустите `scripts/train_yolo.py` с вашим датасетом.

Вклад приветствуется! Откройте issue или PR.

## 📄 Лицензия
MIT License - см. [LICENSE](LICENSE) файл.

## 📬 Контакты
- Автор: [kiryall](https://github.com/kiryall)
- Email: kiryall@yandex.ru
- Issues: [GitHub Issues](https://github.com/kiryall/safer-detection-ocr/issues)

> 🌟 **Для исследователей**: Если вы используете в экспедициях, поделитесь отзывами для улучшения!
