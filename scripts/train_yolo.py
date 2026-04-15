# scripts/train_yolo.py

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

from configs.config import DATA_YAML, YOLO_PRETRAINED_MODEL
from src.utils.logger import setup_logging
from src.utils.yaml_loader import load_config

# Константы и пути
BASE_DIR = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--data", type=str, default=None, help="Path to data.yaml")
args = parser.parse_args()

# Конфигурация и логирование
cfg = load_config(args.config)
logger = setup_logging(log_file="train_yolo.log", remove_file=True)

project_dir = BASE_DIR / "runs" / "seg" / cfg["experiment"]["tag"]

def train_yolo():
    if not YOLO_PRETRAINED_MODEL.exists():
        raise FileNotFoundError(f"Model not found: {YOLO_PRETRAINED_MODEL}")

    model = YOLO(str(YOLO_PRETRAINED_MODEL))
    logger.info(f"Loaded YOLO model from {YOLO_PRETRAINED_MODEL}")

    exp_name = f"{cfg['experiment']['name']}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    data_path = args.data if args.data else str(DATA_YAML)

    logger.info(f"Starting training with experiment name: {exp_name}")
    logger.info(f"Using data: {data_path}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Device: {cfg['train'].get('device', 0)}")

    try:
        results = model.train(
            data=data_path,
            epochs=cfg["train"]["epochs"],
            imgsz=cfg["train"]["imgsz"],
            batch=cfg["train"]["batch"],
            device = cfg["train"].get("device", 0),
            project=str(project_dir.resolve()),
            name=exp_name,
            cache=True,
            workers=2,
            exist_ok=True,
            seed=42,
            patience=10,
        )

        # Сохранение конфигурации в папке с результатами
        save_dir = Path(results.save_dir)        
        shutil.copy(args.config, save_dir / "model_config.yaml")

        logger.info(f"Training completed. Results saved to {save_dir}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    # python -m scripts.train_yolo --config configs/model_config.yaml
    train_yolo()
