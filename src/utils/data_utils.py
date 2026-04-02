# src/utils/data_utils.py
import os
from pathlib import Path
import random
from configs.config import RAW_DATA_DIR


def prep_train_dataset(train_dir: Path,
                  raw_data_dir: Path = RAW_DATA_DIR,
                  num_images: int = 300) -> None:
    """
    Функуция выбирает {num_images} случайных изображений из папки с обучающим датасетом и сохраняет их в папку для тестирования.
    """
    
    # Получаем список всех изображений в папке RAW_DATA_DIR
    all_images = list(raw_data_dir.glob("*.jpg"))
    # Выбираем {num_images} случайных изображений
    selected_images = random.sample(all_images, num_images)
    # копируем выбранные изображения в папку TRAIN_DIR
    try:
        for image in selected_images:
            destination = train_dir / image.name
            os.makedirs(train_dir, exist_ok=True)
            with open(image, 'rb') as src_file:
                with open(destination, 'wb') as dst_file:
                    dst_file.write(src_file.read())
    except Exception as e:
        print(f"Error copying files: {e}")


if __name__ == "__main__":
    prep_train_dataset()